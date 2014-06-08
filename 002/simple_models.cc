
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/logstream.h>

#include <deal.II/lac/full_matrix.h>

#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

//for adaptive meshes - hanging nodes must be taken care of
#include <deal.II/lac/constraint_matrix.h>

//input/output of grid
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h> 
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>


//for adaptive refinement
#include <deal.II/grid/grid_refinement.h>
//for estimating error
#include <deal.II/numerics/error_estimator.h>

//output
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#define _USE_MATH_DEFINES       //we are using M_PI
#include <cmath>

#include "simple_models.hh"
#include "well.hh"
#include "data_cell.hh"
#include "adaptive_integration.hh"
#include "system.hh"


XModel_simple::XModel_simple () : XModel::XModel()
{
}


XModel_simple::XModel_simple (Well* well, 
                              const std::string &name,
                              const unsigned int &n_aquifers) 
  : XModel::XModel(name,n_aquifers)
{
  //we will have array of wells with one well
  wells.push_back(well);
}


XModel_simple::~XModel_simple()
{
}

void XModel_simple::make_grid()
{
  dof_handler->clear();
  coarse_tria.clear();
  
  if(triangulation != NULL)
  {
    triangulation->clear();
    triangulation->clear_flags();
  }
  
  switch (grid_create)
  {
    case load:
    {
        MASSERT(coarse_grid_file != "", "Undefined mesh file!");
        //open filestream with mesh from GMSH
        std::ifstream in;
        GridIn<2> gridin;
        in.open(coarse_grid_file);
        //attaching object of triangulation
        gridin.attach_triangulation(coarse_tria);
        if(in.is_open())
        {
          //reading data from filestream
          gridin.read_msh(in);
        }          
        else
        {
          xprintf(Err, "Could not open coarse grid file: %s", coarse_grid_file.c_str());
        }
        
        triangulation = new PersistentTriangulation<2>(coarse_tria);
        in.close();
        in.clear();
        in.open(ref_flags_file);
        if(in.is_open())
          triangulation->read_flags(in);
        else
        {
          xprintf(Warn, "Could not open refinement flags file: %s\n Ingore this if loading mesh without refinement flag file.\n", ref_flags_file.c_str());
        }
        //creates actual grid to be available
        triangulation->restore();
        break;
    }
    case load_circle:
    {
      GridGenerator::hyper_ball<2>(coarse_tria,center,radius);
      static const HyperBallBoundary<2> boundary(center,radius);
      coarse_tria.set_boundary(0, boundary);
        
      triangulation = new PersistentTriangulation<2>(coarse_tria);
      std::ifstream in;
      in.open(ref_flags_file);
      if(in.is_open())
        triangulation->read_flags(in);
      else
      {
        xprintf(Err, "Could not open refinement flags file: %s\n", ref_flags_file.c_str());
      }
      //creates actual grid to be available
      triangulation->restore();
      break;
    }
    case circle:
    {
      Point<2> temp_center((down_left+up_right)/2);
      double temp_radius =  down_left.distance(up_right) / 2;
      GridGenerator::hyper_ball<2>(coarse_tria,temp_center,temp_radius);
      static const HyperBallBoundary<2> boundary(temp_center,temp_radius);
      coarse_tria.set_boundary(0, boundary);
      //coarse_tria.refine_global (init_refinement);
      break;
    }
    case rect:
    default:
    {
      //square grid
      GridGenerator::hyper_rectangle<2>(coarse_tria, down_left, up_right);
      coarse_tria.refine_global (init_refinement); 
    }
  }
  
  if(grid_create != load &&  grid_create != load_circle)
  {
    //creates persistent triangulation
    triangulation = new PersistentTriangulation<2>(coarse_tria);
    //creates actual grid to be available
    triangulation->restore();
    //MESH OUTPUT - coarse grid = (refinement flags written in output)
    std::stringstream filename1;
    filename1 << output_dir << "coarse_grid.msh";
 
    std::ofstream output (filename1.str());
  
    GridOut grid_out;
    grid_out.write_msh<2> (coarse_tria, output);
    std::cout << "Coarse grid written in file: " << filename1.str() << std::endl;
  }
  
  //MESH OUTPUT 
    std::stringstream filename1;
    filename1 << output_dir << "grid.msh";
 
    std::ofstream output (filename1.str());
  
    GridOut grid_out;
    grid_out.write_msh<2> (*triangulation, output);
    std::cout << "Coarse grid written in file: " << filename1.str() << std::endl;
  
  triangulation_changed = true;
}

void XModel_simple::refine_grid()
{
  // assuming all cells in enrichment area are of the same refinement level
  // and also of the same size in this particular case
  double enriched_cell_diameter = 0;
  double enrichment_radius = rad_enr;

  if(xdata.size() > 0)
    enriched_cell_diameter = xdata[0]->get_cell()->diameter();
  if(r_enr.size() > 0)
    enrichment_radius = r_enr[0];
  
  triangulation->set_all_refine_flags();
  
  if(grid_create == load_circle)
  {  
    Triangulation<2>::active_cell_iterator  
      cell = triangulation->begin_active(),
      endc = triangulation->end();
    //DBGMSG("refinement: %d\n",ref);
    for (; cell!=endc; ++cell)
    {
      if(cell->at_boundary() || (cell->level() > xdata[0]->get_cell()->level()))
      //if(cell->at_boundary() || (cell->center().distance(center) > (enrichment_radius+enriched_cell_diameter) && (cell->level() > 4)))
      //if(cell->center().distance(center) > (r_enr[0]+enriched_cell_diameter)) //cell->at_boundary())
      {
        cell->clear_refine_flag();
      }
    }
  }
  
  triangulation->execute_coarsening_and_refinement();
  triangulation_changed = true;
}



void XModel_simple::assemble_dirichlet()
{
  MASSERT(dirichlet_function != NULL, "Dirichlet BC function has not been set.\n");
  MASSERT(dof_handler != NULL, "DoF Handler object does not exist.\n");
  /*
   //Definition of diferent types of BC
    typename Triangulation<2>::cell_iterator
             cell = triangulation->begin (),
             endc = triangulation->end();
             
             for (; cell!=endc; ++cell)
             {
               //if (!cell->at_boundary() ) continue;
               unsigned int n_boundaries = 0;
               for (unsigned int face=0;
                    face<GeometryInfo<2>::faces_per_cell;
                    ++face)
               {
                   if(cell->face(face)->at_boundary()) n_boundaries++;
                       
//                 if ( (cell->face(face)->center()(1) - down_left[1] < 1e-10) 
//                    )
//                   cell->face(face)->set_boundary_indicator (1);
               }
               
               if(n_boundaries > 1) 
               {
                   for (unsigned int face=0;
                    face<GeometryInfo<2>::faces_per_cell;
                    ++face)
                    {
                        if(cell->face(face)->at_boundary())
                            cell->face(face)->set_boundary_indicator (1);
                    }
               }
             }
  //*/

  std::map<unsigned int,double> boundary_values;
  VectorTools::interpolate_boundary_values (*dof_handler,
                                            0,
                                            //XModel_simple::Dirichlet_pressure(wells[0]),
                                            *dirichlet_function,
                                            boundary_values);
  
    /*
    // Setting the enriched part boundary condition
    ExactBase * exact_solution = static_cast<ExactSolution*>( dirichlet_function );
  
    XDataCell * local_xdata;
    unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);   
    
    DoFHandler<2>::active_cell_iterator
        cell = dof_handler->begin_active(),
        endc = dof_handler->end();
    for (; cell!=endc; ++cell)
    {
        //DBGMSG("cell: %d\n",cell->index());
        // is there is NOT a user pointer on the cell == is not enriched?
        
        if(cell->at_boundary())
        if (cell->user_pointer() != nullptr)
        {   
            local_xdata = static_cast<XDataCell*>( cell->user_pointer() );
            
            for(unsigned int face_no=0; face_no < GeometryInfo<2>::faces_per_cell; face_no++)
            {
                if(cell->face(face_no)->at_boundary())
                {
                    for(unsigned int j = 0; j < GeometryInfo<2>::vertices_per_face; j++) //M_w
                    {
                        unsigned int k = GeometryInfo<2>::face_to_cell_vertices(face_no,j); //cell vertex index
                        for(unsigned int w = 0; w < local_xdata->n_wells(); w++) //W
                        {
                            if(local_xdata->global_enriched_dofs(w)[k] != 0)
                                boundary_values[local_xdata->global_enriched_dofs(w)[k]] = exact_solution->a();
                        }
                    }
                }
            }
        }
    }
    //*/
    
   /*
   //if boundary elements are enriched, we must set the enrichment dofs
   //iterator over cells
   DoFHandler<2>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  
  XData *loc_xdata;
  std::vector<unsigned int> local_vertex_indexes;
  std::pair<unsigned int, double> enrich_pair;
   
  // face and vertex numbering - DEAL II
  //       3
  //    2-->--3
  //    |     |
  //   0^     ^1
  //    |     |
  //    0-->--1
  //       2
  
  
  for (; cell!=endc; ++cell)
    {
      if (!cell->at_boundary() ) continue;
      if (cell->user_pointer() == NULL ) continue;
      
      local_vertex_indexes.clear();
      for (unsigned int face=0;
           face<GeometryInfo<2>::faces_per_cell;
           ++face)
        {
          if(cell->face(face)->at_boundary())
          {
            for(unsigned int i=0; i < fe.dofs_per_face; i++)
            {
              local_vertex_indexes.push_back(GeometryInfo<2>::face_to_cell_vertices(face,i));
            }
          }
        }
      
      loc_xdata = static_cast<XData*>( cell->user_pointer() );
      
      for(unsigned int k=0; k < local_vertex_indexes.size(); k++)
      {
        for (unsigned int w=0; w < loc_xdata->wells().size(); w++)
        {
          if(loc_xdata->global_enriched_dofs()[w][local_vertex_indexes[k]] != 0)
          {
            unsigned int dof = loc_xdata->global_enriched_dofs()[w][local_vertex_indexes[k]];
            enrich_pair = std::make_pair<unsigned int&, double>(dof, 0.0 );
            
            boundary_values.insert(enrich_pair);
          }
        }
      }
    }
  */
   DBGMSG("boundary_values size = %d\n",boundary_values.size());
   MatrixTools::apply_boundary_values (boundary_values,
                                       block_matrix,
                                       block_solution,
                                       block_system_rhs,
                                       true
                                      );
   
   std::cout << "Dirichlet BC assembled succesfully." << std::endl;
}




Model_simple::Model_simple () : Model::Model()
{
}


Model_simple::Model_simple (Well* well,
                            const std::string &name,
                            const unsigned int &n_aquifers) 
  : Model::Model(name, n_aquifers)
{
  //we will have array of wells with one well
  wells.push_back(well);
  
  //TODO: check if all wells lie in the area
}


Model_simple::~Model_simple()
{
}



void Model_simple::refine_grid()
{
  //vector for the estimated error on each cell
  Vector<float> estimated_error_per_cell (triangulation->n_active_cells());
  
  //computes error
  KellyErrorEstimator<2>::estimate (*dof_handler,
                                    QGauss<1>(3),
                                    typename FunctionMap<2>::type(),
                                    block_solution.block(0),
                                    estimated_error_per_cell);
  
  //setting flags for refinement and coarsing
  GridRefinement::refine_and_coarsen_fixed_number (*triangulation,
                                                   estimated_error_per_cell,
                                                   refinement_percentage, 
                                                   coarsing_percentage);
  
  if(grid_create == load_circle)
  {
    Triangulation<2>::active_cell_iterator  
      cell = triangulation->begin_active(),
      endc = triangulation->end();
    //DBGMSG("refinement: %d\n",ref);
    for (; cell!=endc; ++cell)
    {
      if( cell->at_boundary() || (cell->center().distance(center) > radius/2.0 && cell->level() > 4) )
      {
        cell->clear_refine_flag();
      }
    }
  }
  
  //doing refinement and coarsing
  triangulation->execute_coarsening_and_refinement ();
  
  triangulation_changed = true;
  
  std::cout << "Number of active cells:       "
            << triangulation->n_active_cells()
            << std::endl;
  std::cout << "Total number of cells: "
            << triangulation->n_cells()
            << std::endl;
            
}



void Model_simple::assemble_dirichlet()
{
  MASSERT(dirichlet_function != NULL, "Dirichlet BC function has not been set.\n");
  MASSERT(dof_handler != NULL, "DoF Handler object does not exist.\n");
  /* //Definition of diferent types of BC
  typename Triangulation<2>::cell_iterator
             cell = triangulation->begin (),
             endc = triangulation->end();
             
             for (; cell!=endc; ++cell)
             {
               //if (!cell->at_boundary() ) continue;
               for (unsigned int face=0;
                    face<GeometryInfo<2>::faces_per_cell;
                    ++face)
               {
                
                if ( (cell->face(face)->center()(1) - down_left[1] < 1e-10) 
                   )
                  cell->face(face)->set_boundary_indicator (1);
               }
             }
  */
             
  std::map<unsigned int,double> boundary_values;
   VectorTools::interpolate_boundary_values (*dof_handler,
                                             0,
                                             //Model_simple::Dirichlet_pressure(wells[0]),
                                             *dirichlet_function,
                                             boundary_values);
  
   MatrixTools::apply_boundary_values (boundary_values,
                                       block_matrix.block(0,0),
                                       block_solution.block(0),
                                       block_system_rhs.block(0));
   
   std::cout << "Dirichlet BC assembled succesfully." << std::endl;
}
                               
