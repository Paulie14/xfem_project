
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/logstream.h>

#include <deal.II/lac/full_matrix.h>

#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/iterative_inverse.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/eigen.h>

#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/base/function.h>

//for adaptive meshes - hanging nodes must be taken care of
#include <deal.II/lac/constraint_matrix.h>

//input/output of grid
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h> 
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/persistent_tria.h>


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

#include "model_base.hh"
#include "xmodel.hh"
#include "well.hh"
#include "data_cell.hh"
#include "adaptive_integration.hh"
#include "system.hh"


XModel::XModel () 
  : Model_base(),
    enrichment_method_(Enrichment_method::xfem_shift),
    well_computation_(Well_computation::bc_newton),
    rad_enr(0),
    n_enriched_dofs(0),
    //dealii fem
    triangulation(NULL),
    fe (1),
    quadrature_formula(2),
    fe_values (fe, quadrature_formula,
      update_values | update_gradients | update_JxW_values),
    hanging_nodes(true),
    out_decomposed(true),
    out_shape_functions(false)
{
  name = "Default_XFEM_Model";
  dof_handler = new DoFHandler<2>();
}

XModel::XModel (const std::string &name, 
                const unsigned int &n_aquifers) 
:   Model_base::Model_base(name, n_aquifers),
    enrichment_method_(Enrichment_method::xfem_shift),
    well_computation_(Well_computation::bc_newton),
    rad_enr(0),
    n_enriched_dofs(0),
    //dealii fem
    triangulation(NULL),
    fe (1),
    quadrature_formula(2),
    fe_values (fe, quadrature_formula,
      update_values | update_gradients | update_JxW_values),
    hanging_nodes(true),
    out_decomposed(true),
    out_shape_functions(false)
    
{
  dof_handler = new DoFHandler<2>();
}

XModel::XModel (const std::vector<Well*> &wells, 
                const std::string &name, 
                const unsigned int &n_aquifers) 
:   Model_base::Model_base(wells, name, n_aquifers),
    enrichment_method_(Enrichment_method::xfem_shift),
    well_computation_(Well_computation::bc_newton),
    rad_enr(0),
    n_enriched_dofs(0),
    //dealii fem
    triangulation(NULL),
    fe (1),
    quadrature_formula(2),
    fe_values (fe, quadrature_formula,
      update_values | update_gradients | update_JxW_values),
    hanging_nodes(true),
    out_decomposed(true),
    out_shape_functions(false)
    
{
  //DBGMSG("XModel constructor, wells_size: %d\n",this->wells.size());
  dof_handler = new DoFHandler<2>();
  r_enr.resize(wells.size());
}


XModel::~XModel()
{
  for(unsigned int i=0; i < xdata.size(); i++)
    delete xdata[i];
  
  if(dof_handler != NULL)
    delete dof_handler;
  
  if(triangulation != NULL)
    delete triangulation;
}


void XModel::make_grid ()
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
        xprintf(Err, "Not implemented.");
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
  
  std::cout << "Number of active cells:       "
            << triangulation->n_active_cells()
            << std::endl;
  std::cout << "Total number of cells: "
            << triangulation->n_cells()
            << std::endl;
  triangulation_changed = true;
}

void XModel::refine_grid()
{
  triangulation->refine_global(1);
  triangulation_changed = true;
  
  std::cout << "Number of active cells:       "
            << triangulation->n_active_cells()
            << std::endl;
  std::cout << "Total number of cells: "
            << triangulation->n_cells()
            << std::endl;
}


//------------------------------------------------------------------------------ FIND ENRICHED CELLS
void XModel::find_enriched_cells()
{
  DBGMSG("find_enriched_cells: wells_size: %d\n",wells.size());
  MASSERT(wells.size() > 0, "No wells are defined in vector of wells");
  
  r_enr.resize(wells.size());
   
  //iterator over cells
  DoFHandler<2>::active_cell_iterator cell,endc;

  //tells us at which global index new enriched dof should be obtained
  unsigned int n_global_enriched_dofs = dof_handler->n_dofs();
  
  std::vector<unsigned int> local_dof_indices(fe.dofs_per_cell);
  
  //on each cell iteration over all wells
  for (unsigned int w = 0; w < wells.size(); w++)
  {
    //temporary vector of enriched dofs: i-th dof is enriched by dof dof_number[i]
    std::vector<unsigned int> enriched_dof_indices; 
    std::vector<unsigned int> enriched_weights;
    //all are initially zero (zero dof is definitely one of the regular dof, so zero means unenriched)
    enriched_dof_indices.resize(dof_handler->n_dofs(),0); 
    enriched_weights.resize(dof_handler->n_dofs(),0); //0 is for unenriched dof
  
    //finding the cell in which the center lies
    //adding first quadrature points (circle integration)
    cell = dof_handler->begin_active();
    endc = dof_handler->end();
    for (; cell!=endc; ++cell)
    { 
      if ( cell->point_inside(wells[w]->center()) )
      {
        DBGMSG("enrich well: %d from cell: %d\n", w, cell->index());
        
        //getting r_enr (circle in which dofs(nodes) will be enriched)
        //setting first XData for the cell in which the center of the well lies
        r_enr[w]=wells[w]->radius(); //can be set to well radius or 0
        r_enr[w] = std::max(rad_enr, r_enr[w]);
        
        //check off the minimal enrichment radius (has to include at least one node)
        double dist = std::max(cell->vertex(0).distance(wells[w]->center()), r_enr[w]);
        for(unsigned int i=1; i < fe.dofs_per_cell; i++)
        {
          //assumming that the well radius is smaller than the cell
          dist = std::min(cell->vertex(i).distance(wells[w]->center()), r_enr[w]);
          //dist = std::max(cell->vertex(i).distance(wells[w]->center()), dist);
        }  
        r_enr[w] = std::max(dist, r_enr[w]);
          
        triangulation->clear_user_flags();
        
        if(enrichment_method_ == Enrichment_method::sgfem)
          enrich_cell_sgfem(cell, w, enriched_dof_indices, n_global_enriched_dofs);
        else
          enrich_cell(cell, w, enriched_dof_indices, enriched_weights, n_global_enriched_dofs);
        
        std::cout << "enrichment radius: wanted: " << rad_enr << "  finally set: " << r_enr[w] << std::endl;
        
        break;
      }
    } //for cells
  }
  
  
  n_enriched_dofs = n_global_enriched_dofs - dof_handler->n_dofs();
  std::cout << "Number of enriched dofs: " << n_enriched_dofs << std::endl;
  
  
  //printing enriched nodes and dofs
  
  DBGMSG("Printing xdata:\n");
  for(unsigned int i=0; i < xdata.size(); i++)
  {
    std::cout << "(" << setw(4) << i << ") ";
    for(unsigned int xw=0; xw < xdata[i]->n_wells(); xw++)
    {
      std::cout << " w=" << setw(3) << xw << " well center: " << setw(7) << xdata[i]->get_well(xw)->center() << "\tglobal_enrich_dofs: [";
      
      for(unsigned int j=0; j < fe.dofs_per_cell; j++)
        std::cout << std::setw(4) << xdata[i]->global_enriched_dofs(xw)[j] << "  ";
      std::cout << "]   [";
      
      for(unsigned int j=0; j < fe.dofs_per_cell; j++)
        std::cout << std::setw(4) << xdata[i]->weights(xw)[j] << "  ";
      std::cout << "]";
      
      std::cout << "  n_qpoints=" <<  xdata[i]->q_points(xw).size() << std::endl;
    }
  }
  //*/
}


//-------------------------------------------------------------------------------------- ENRICH CELL
void XModel::enrich_cell ( const DoFHandler<2>::active_cell_iterator cell,
                           const unsigned int &well_index,
                           std::vector<unsigned int> &enriched_dof_indices,
                           std::vector<unsigned int> &enriched_weights,
                           unsigned int &n_global_enriched_dofs
                         )
{
  //std::cout << "CALL enrich_cell on: " << cell->index();
  // if the flag is set = we have been already there, so continue
    if ( cell->user_flag_set() == true)
    {
      //std::cout << "\thave already been at cell\n";
      return;
    }
  
  //sets user flag for the cell in which we have been
  cell->set_user_flag();
  
  //flag is true if no node is to be enriched (is not in the enrichment radius )
  bool cell_not_enriching = true;
  
  for(unsigned int i=0; i < fe.dofs_per_cell; i++)
  {
    if(cell->vertex(i).distance(wells[well_index]->center()) <= r_enr[well_index])
    {
      cell_not_enriching = false;
    }
  }
  
  //if there is no previously enriched dof we can return
  if (cell_not_enriching)
  {
    return;
  }
  //else
  
  
  std::vector<unsigned int> local_dof_indices(fe.dofs_per_cell);
  cell->get_dof_indices(local_dof_indices);
  
  //else we continue by enriching dofs
  //std::cout << "\tenriched: ";
  //temporary space for enriched dofs
  std::vector<unsigned int> local_enriched_dofs(fe.dofs_per_cell,0);
  std::vector<unsigned int> local_enriched_node_weights(fe.dofs_per_cell,0);
  //enriching dofs
  for(unsigned int i=0; i < fe.dofs_per_cell; i++)
  {
    //if the node was previously enriched
    if (enriched_dof_indices[local_dof_indices[i]] != 0)
    {
      //adding enriched dof that has been already defined from previous cell
      local_enriched_dofs[i] = enriched_dof_indices[local_dof_indices[i]];
      local_enriched_node_weights[i] = enriched_weights[local_dof_indices[i]];
    }
    else
    { 
      //if the node should be enriched
      if (cell->vertex(i).distance(wells[well_index]->center()) <= r_enr[well_index])
      {
        //if the node is from index subset N_w then the weight is 1 (else zero on blending element)
        local_enriched_node_weights[i] = 1;
        enriched_weights[local_dof_indices[i]] = 1;
      }
      
      //adding degree of freedom for every node on enriched element (both reproducing and blending elements)
      enriched_dof_indices[local_dof_indices[i]] = n_global_enriched_dofs;
      local_enriched_dofs[i] = n_global_enriched_dofs;
      n_global_enriched_dofs ++;
    }
    //std::cout << enriched_dof_indices[local_dof_indices[i]] << " ";
  }
  //std::cout << std::endl;
 
 
  /////-----------------------------Well Boundary Integration part--------------start
  //if(well_computation_ == Well_computation::bc_newton)
  {
    //looking for quadrature poitns of the well in the current cell
    //temporary vector of q_points
    std::vector<const Point<2>* > points;
    //flag for addition cell to the vector in well object
    bool q_points_to_add = false;
    //checking if the q_points are in the cell
    for(unsigned int p=0; p < wells[well_index]->q_points().size(); p++)
    {
      if (cell->point_inside(wells[well_index]->q_points()[p]))
      {
        q_points_to_add = true;
        //adding point
        points.push_back( &(wells[well_index]->q_points()[p]) );
      }
    }
    
    //checking if there has been xdata already created
    //if not it is created
    //if so it means that XData object has been created and there are additions from other wells
    if(cell->user_pointer() == NULL)
    {
      if(q_points_to_add)
      xdata.push_back(new XDataCell(cell, 
                                    wells[well_index], 
                                    well_index, 
                                    local_enriched_dofs, 
                                    local_enriched_node_weights, 
                                    points));
      else
      xdata.push_back(new XDataCell(cell, 
                                    wells[well_index], 
                                    well_index, 
                                    local_enriched_dofs, 
                                    local_enriched_node_weights));
    
      cell->set_user_pointer(xdata.back());
    }
    else
    {
      XDataCell* xdata_pointer = static_cast<XDataCell*> (cell->user_pointer());
      if(q_points_to_add)
        xdata_pointer->add_data(wells[well_index], well_index, 
                                local_enriched_dofs, local_enriched_node_weights, points );
      else
        xdata_pointer->add_data(wells[well_index], well_index, 
                                local_enriched_dofs, local_enriched_node_weights );
    }
  }
  /////-----------------------------Well Boundary Integration part--------------end
  /*
  //this will be possible if we know at this point whether the well crosses the cell (we know this only from q_points)
  else if(well_computation_ == Well_computation::sources)
  {
    //checking if there has been xdata already created
    //if not it is created
    //if so it means that XData object has been created and there are additions from other wells
    if(cell->user_pointer() == NULL)
    {
      xdata.push_back(new XDataCell(cell, 
                                    wells[well_index], 
                                    well_index, 
                                    local_enriched_dofs, 
                                    local_enriched_node_weights));
    
      cell->set_user_pointer(xdata.back());
    }
    else
    {
      XDataCell* xdata_pointer = static_cast<XDataCell*> (cell->user_pointer());
        xdata_pointer->add_data(wells[well_index], well_index, 
                                local_enriched_dofs, local_enriched_node_weights );
    }
  }
  //*/
  
  // searching neighbors...
  for (unsigned int face_no=0; face_no < GeometryInfo<2>::faces_per_cell; ++face_no)
  {
    //std::cout << "\tface(" << face_no << "): ";
    typename DoFHandler<2>::face_iterator face = cell->face(face_no);
    
    //if the face is at the boundary, there is no neighbor, so continue
    if (face->at_boundary()) 
    {
      //std::cout << "at the boundary\n";
      continue;
    }
    
    // asking face about children - if so, then there must also be finer cells 
    // which are children or farther offsprings of our neighbor.
    if (face->has_children())
      {
        //std::cout << "\t Face has children." << std::endl;
        //TODO: check if adaptivity
        // iteration over subfaces - children
        for (unsigned int subface_no = 0; subface_no < face->number_of_children(); ++subface_no)
          {
            typename DoFHandler<2>::cell_iterator neighbor_child
                           = cell->neighbor_child_on_subface (face_no, subface_no);
            Assert (!neighbor_child->has_children(), ExcInternalError());
            
            //std::cout << "\tFace=" << face_no << " subface=" << subface_no 
            //          << " entering cell=" << neighbor_child->index() << std::endl;
            // entering on the neighbor's children behind the current face
            enrich_cell(neighbor_child, well_index, enriched_dof_indices, enriched_weights, n_global_enriched_dofs);
          }
          //*/
      }
    else
      {
        // creating neighbor's cell iterator
        MASSERT(cell->neighbor(face_no).state() == IteratorState::valid,
            "Neighbor's state is invalid.");
        typename DoFHandler<2>::cell_iterator neighbor = cell->neighbor(face_no);
        MASSERT(!neighbor->has_children(), "Neighbor has children and is not active!");
        
        // asking if the neighbor is coarser, if not then it is neither coarser nor finer
        // so it is the same level of refinement
        if (!cell->neighbor_is_coarser(face_no) 
            //&&
            //(neighbor->index() > cell->index() ||
            //(neighbor->level() < cell->level() &&
            //neighbor->index() == cell->index() ))
           )
          {
            //std::cout << "\tneigh: " << neighbor->index() << "\t same refine level: " 
            //          << neighbor->level() << std::endl;
            enrich_cell(neighbor, well_index, enriched_dof_indices, enriched_weights, n_global_enriched_dofs);
          } 
        else
          //is coarser
          {
            //std::cout << "\tneigh: " << neighbor->index() << "\t is coarser" << std::endl; 
            enrich_cell(neighbor, well_index, enriched_dof_indices, enriched_weights, n_global_enriched_dofs);
          }
      }
  }
}

//-------------------------------------------------------------------------------------- ENRICH CELL
void XModel::enrich_cell_sgfem ( const DoFHandler<2>::active_cell_iterator cell,
                           const unsigned int &well_index,
                           std::vector<unsigned int> &enriched_dof_indices,
                           unsigned int &n_global_enriched_dofs
                         )
{
  //std::cout << "CALL enrich_cell on: " << cell->index();
  // if the flag is set = we have been already there, so continue
    if ( cell->user_flag_set() == true)
    {
      //std::cout << "\thave already been at cell\n";
      return;
    }
  
  //sets user flag for the cell in which we have been
  cell->set_user_flag();
  
  //flag is true if no node is to be enriched (is not in the enrichment radius )
  bool cell_not_enriching = true;
  
  for(unsigned int i=0; i < fe.dofs_per_cell; i++)
  {
    if(cell->vertex(i).distance(wells[well_index]->center()) <= r_enr[well_index])
    {
      cell_not_enriching = false;
    }
  }
  
  //if there is no previously enriched dof we can return
  if (cell_not_enriching)
  {
    return;
  }
  //else
  
  
  std::vector<unsigned int> local_dof_indices(fe.dofs_per_cell);
  cell->get_dof_indices(local_dof_indices);
  
  //else we continue by enriching dofs
  //std::cout << "\tenriched: ";
  //temporary space for enriched dofs
  std::vector<unsigned int> local_enriched_dofs(fe.dofs_per_cell,0);
  std::vector<unsigned int> local_enriched_node_weights(fe.dofs_per_cell,0);    //not needed - is input to XDataCell::add(...)
  //enriching dofs
  for(unsigned int i=0; i < fe.dofs_per_cell; i++)
  {
    //if the node was previously enriched
    if (enriched_dof_indices[local_dof_indices[i]] != 0)
    {
      //adding enriched dof that has been already defined from previous cell
      local_enriched_dofs[i] = enriched_dof_indices[local_dof_indices[i]];
    }
    else
    { 
      //if the node should be enriched
      if (cell->vertex(i).distance(wells[well_index]->center()) <= r_enr[well_index])
      {
        //adding degree of freedom for every (and only) ENRICHED node on enriched element
        enriched_dof_indices[local_dof_indices[i]] = n_global_enriched_dofs;
        local_enriched_dofs[i] = n_global_enriched_dofs;
        n_global_enriched_dofs ++;
      }
    }
    //std::cout << enriched_dof_indices[local_dof_indices[i]] << " ";
  }
  //std::cout << std::endl;
 
 
  /////-----------------------------Well Boundary Integration part--------------start
  //if(well_computation_ == Well_computation::bc_newton)
  {
    //looking for quadrature poitns of the well in the current cell
    //temporary vector of q_points
    std::vector<const Point<2>* > points;
    //flag for addition cell to the vector in well object
    bool q_points_to_add = false;
    //checking if the q_points are in the cell
    for(unsigned int p=0; p < wells[well_index]->q_points().size(); p++)
    {
      if (cell->point_inside(wells[well_index]->q_points()[p]))
      {
        q_points_to_add = true;
        //adding point
        points.push_back( &(wells[well_index]->q_points()[p]) );
      }
    }
    
    //checking if there has been xdata already created
    //if not it is created
    //if so it means that XData object has been created and there are additions from other wells
    if(cell->user_pointer() == NULL)
    {
      if(q_points_to_add)
      xdata.push_back(new XDataCell(cell, 
                                    wells[well_index], 
                                    well_index, 
                                    local_enriched_dofs, 
                                    local_enriched_node_weights, 
                                    points));
      else
      xdata.push_back(new XDataCell(cell, 
                                    wells[well_index], 
                                    well_index, 
                                    local_enriched_dofs, 
                                    local_enriched_node_weights));
    
      cell->set_user_pointer(xdata.back());
    }
    else
    {
      XDataCell* xdata_pointer = static_cast<XDataCell*> (cell->user_pointer());
      if(q_points_to_add)
        xdata_pointer->add_data(wells[well_index], well_index, 
                                local_enriched_dofs, local_enriched_node_weights, points );
      else
        xdata_pointer->add_data(wells[well_index], well_index, 
                                local_enriched_dofs, local_enriched_node_weights );
    }
  }
  /////-----------------------------Well Boundary Integration part--------------end
  /*
  //this will be possible if we know at this point whether the well crosses the cell (we know this only from q_points)
  else if(well_computation_ == Well_computation::sources)
  {
    //checking if there has been xdata already created
    //if not it is created
    //if so it means that XData object has been created and there are additions from other wells
    if(cell->user_pointer() == NULL)
    {
      xdata.push_back(new XDataCell(cell, 
                                    wells[well_index], 
                                    well_index, 
                                    local_enriched_dofs, 
                                    local_enriched_node_weights));
    
      cell->set_user_pointer(xdata.back());
    }
    else
    {
      XDataCell* xdata_pointer = static_cast<XDataCell*> (cell->user_pointer());
        xdata_pointer->add_data(wells[well_index], well_index, 
                                local_enriched_dofs, local_enriched_node_weights );
    }
  }
  //*/
  
  // searching neighbors...
  for (unsigned int face_no=0; face_no < GeometryInfo<2>::faces_per_cell; ++face_no)
  {
    //std::cout << "\tface(" << face_no << "): ";
    typename DoFHandler<2>::face_iterator face = cell->face(face_no);
    
    //if the face is at the boundary, there is no neighbor, so continue
    if (face->at_boundary()) 
    {
      //std::cout << "at the boundary\n";
      continue;
    }
    
    // asking face about children - if so, then there must also be finer cells 
    // which are children or farther offsprings of our neighbor.
    if (face->has_children())
      {
        //std::cout << "\t Face has children." << std::endl;
        //TODO: check if adaptivity
        // iteration over subfaces - children
        for (unsigned int subface_no = 0; subface_no < face->number_of_children(); ++subface_no)
          {
            typename DoFHandler<2>::cell_iterator neighbor_child
                           = cell->neighbor_child_on_subface (face_no, subface_no);
            Assert (!neighbor_child->has_children(), ExcInternalError());
            
            //std::cout << "\tFace=" << face_no << " subface=" << subface_no 
            //          << " entering cell=" << neighbor_child->index() << std::endl;
            // entering on the neighbor's children behind the current face
            enrich_cell_sgfem(neighbor_child, well_index, enriched_dof_indices, n_global_enriched_dofs);
          }
          //*/
      }
    else
      {
        // creating neighbor's cell iterator
        MASSERT(cell->neighbor(face_no).state() == IteratorState::valid,
            "Neighbor's state is invalid.");
        typename DoFHandler<2>::cell_iterator neighbor = cell->neighbor(face_no);
        MASSERT(!neighbor->has_children(), "Neighbor has children and is not active!");
        
        // asking if the neighbor is coarser, if not then it is neither coarser nor finer
        // so it is the same level of refinement
        if (!cell->neighbor_is_coarser(face_no) 
            //&&
            //(neighbor->index() > cell->index() ||
            //(neighbor->level() < cell->level() &&
            //neighbor->index() == cell->index() ))
           )
          {
            //std::cout << "\tneigh: " << neighbor->index() << "\t same refine level: " 
            //          << neighbor->level() << std::endl;
            enrich_cell_sgfem(neighbor, well_index, enriched_dof_indices, n_global_enriched_dofs);
          } 
        else
          //is coarser
          {
            //std::cout << "\tneigh: " << neighbor->index() << "\t is coarser" << std::endl; 
            enrich_cell_sgfem(neighbor, well_index, enriched_dof_indices, n_global_enriched_dofs);
          }
      }
  }
}



//------------------------------------------------------------------------------------------- SETUP    
void XModel::setup_system ()
{
  //before using block_sp_pattern again, block_matrix must be cleared
  //to destroy pointer to block_sp_pattern
  //else the copy constructor will destroy block_sp_pattern
  //and block_matrix would point to nowhere!!
  block_matrix = 0.0;
  block_matrix.clear();
  
  for(unsigned int x=0; x < xdata.size(); x++)
    delete xdata[x];
  
  xdata.clear();
  n_enriched_dofs = 0;
  triangulation->clear_user_data();
  //all data from previous run are cleared
  
  
  dof_handler->initialize(*triangulation,fe);
  
  std::cout << "Number of unenriched degrees of freedom: "
	    << dof_handler->n_dofs()
	    << std::endl;

  //find cells which lies within the enrichment radius of the wells
  find_enriched_cells();
  
  
  //adding well dofs to xdata
  for(unsigned int x=0; x < xdata.size(); x++)
  {
    std::vector<unsigned int> well_dof_indices(xdata[x]->n_wells(), 0);
    for(unsigned int w=0; w < xdata[x]->n_wells(); w++)
    {
      //DBGMSG("setup-well_dof_indices: wi=%d \n", xdata[x]->get_well_index(w));
      well_dof_indices[w] = dof_handler->n_dofs() + n_enriched_dofs + xdata[x]->get_well_index(w);
    }
    xdata[x]->set_well_dof_indices(well_dof_indices);
  }
  
  if(hanging_nodes)
  {
    //HANGING NODES
    //clearing after the last refinement cycle
    hanging_node_constraints.clear();
    //making hanging nodes
    DoFTools::make_hanging_node_constraints (*dof_handler,
                                             hanging_node_constraints);
    //finalizing hanging nodes for this dof_handler
    hanging_node_constraints.close();
    //hanging_node_constraints.print(std::cout);
    //std::cout << "number of constrains: " << hanging_node_constraints.n_constraints() << std::endl;
  }
  
  //BLOCK SPARSITY PATTERN
  //inicialization of (temporary) BlockCompressedSparsityPattern, BlockVector
  const unsigned int blocks_dimension = 3;
  unsigned int n[blocks_dimension] = 
                      { dof_handler->n_dofs(), //n1-block(0) unenriched dofs
                        n_enriched_dofs,      //n2-block(1) enriched dofs
                        wells.size()          //n3-block(2) average pressures on wells
                      };
                    
  BlockCompressedSparsityPattern block_c_sparsity(blocks_dimension, blocks_dimension);
  for(unsigned int i=0; i < blocks_dimension; i++)
    for(unsigned int j=0; j < blocks_dimension; j++)
      block_c_sparsity.block(i,j).reinit(n[i],n[j]);
    
  block_c_sparsity.collect_sizes();

      
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  std::vector<unsigned int> local_dof_indices(dofs_per_cell);
  unsigned int n_well_block = n[0] + n[1];
  
  DoFHandler<2>::active_cell_iterator
    cell = dof_handler->begin_active(),
    endc = dof_handler->end();
  for (; cell!=endc; ++cell)
  {
    cell->get_dof_indices(local_dof_indices);
    
    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
      for (unsigned int j=0; j<dofs_per_cell; ++j)
      {
        //block A
        block_c_sparsity.block(0,0).add(local_dof_indices[i],local_dof_indices[j]);
      }
    }
    
    if (cell->user_pointer() != NULL)
    { 
      //A *a=static_cast<A*>(cell->user_pointer()); //from DEALII (TriaAccessor)
      XDataCell *cell_xdata = static_cast<XDataCell*>( cell->user_pointer() );
      
      for(unsigned int w=0; w < cell_xdata->n_wells(); w++)
      {
        for(unsigned int k=0; k < dofs_per_cell; k++)
        {
          {
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              if(cell_xdata->global_enriched_dofs(w)[k] != 0)
              {
                //block R_transpose
                block_c_sparsity.add(local_dof_indices[i], cell_xdata->global_enriched_dofs(w)[k]);
                //block R
                block_c_sparsity.add(cell_xdata->global_enriched_dofs(w)[k], local_dof_indices[i]);
                //block C_transpose - enriched dofs
                block_c_sparsity.add(cell_xdata->global_enriched_dofs(w)[k], n_well_block + cell_xdata->get_well_index(w));
                //block C - enriched dofs
                block_c_sparsity.add(n_well_block + cell_xdata->get_well_index(w), cell_xdata->global_enriched_dofs(w)[k]);
              }
              
              //block S
              for(unsigned int ww=0; ww < cell_xdata->n_wells(); ww++)
              {
                if(cell_xdata->global_enriched_dofs(w)[i] != 0
                   && cell_xdata->global_enriched_dofs(ww)[k] != 0)
                block_c_sparsity.add(cell_xdata->global_enriched_dofs(w)[i], 
                                     cell_xdata->global_enriched_dofs(ww)[k]);
              }
            }
          }
          //block C, C_transpose - unenriched dofs
          if(cell_xdata->q_points(w).size() > 0)        //does the well cross the cell?
          {
            block_c_sparsity.add(local_dof_indices[k], n_well_block + cell_xdata->get_well_index(w));
            block_c_sparsity.add(n_well_block + cell_xdata->get_well_index(w), local_dof_indices[k]);
          }
        }
      }
    }
  }
  
  //diagonal pattern in the block (2,2)
  for(unsigned int w = 0; w < wells.size(); w++)
    block_c_sparsity.block(2,2).add(w,w);
  
  if(hanging_nodes)
  {
    //condensing hanging nodes
    hanging_node_constraints.condense(block_c_sparsity);
  }
  
  //copy from (temporary) BlockCompressedSparsityPattern to (main) BlockSparsityPattern
  block_sp_pattern.copy_from(block_c_sparsity);
 
  //reinitialization of block_matrix
  block_matrix.reinit(block_sp_pattern);
  
  //BLOCK VECTOR - SOLUTION, RHS REINITIALIZATION, filling zeros
  //two blocks in vectors
  block_solution.reinit(blocks_dimension);
  block_system_rhs.reinit(blocks_dimension);
  
  //reinitialization of block_solution 
  //(N,fast=false) .. vector is filled with zeros
  for(unsigned int i=0; i < blocks_dimension; i++)
  {
    block_solution.block(i).reinit(n[i]);
    block_system_rhs.block(i).reinit(n[i]);
  }
  
  block_solution.collect_sizes();
  block_system_rhs.collect_sizes();
  
  
  //prints number of nozero elements in block_c_sparsity
  std::cout << "nozero elements in block_sp_pattern: " << block_sp_pattern.n_nonzero_elements() << std::endl;
  
  if(sparsity_pattern_output_)
  {
    //prints whole BlockSparsityPattern
    std::ofstream out1 (output_dir+"block_sp_pattern.1");
    block_sp_pattern.print_gnuplot (out1);

    //prints SparsityPattern of the block (0,0)
    std::ofstream out2 (output_dir+"00_sp_pattern.1");
    block_sp_pattern.block(0,0).print_gnuplot (out2);
  }
  
  
  //DBGMSG("Printing sparsity pattern: \n");
  //block_sp_pattern.print(std::cout);
  //std::cout << "\n\n";
}



void XModel::assemble_system ()
{
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  //Vector<double>       cell_rhs (dofs_per_cell);	//HOMOGENOUS NEUMANN -> = 0
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  
  //for checking number of active cells (there used to be a problem in adaptivity)
  unsigned int count = 0;
  
  DoFHandler<2>::active_cell_iterator
    cell = dof_handler->begin_active(),
    endc = dof_handler->end();
  for (; cell!=endc; ++cell)
  {
    count++;
    fe_values.reinit (cell);
    cell_matrix = 0;
    //cell_rhs = 0;		//HOMOGENOUS NEUMANN -> = 0
    
    /*
    //printing Jakobi determinants, fe_values flag must be set in constructors: update_jacobians
    //Jakobi determinant is constant for all quadrature points sofar
    DBGMSG("Jakobian unenriched cell(%d): ", cell->index());
    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
    {
      std::cout << "  " << fe_values.jacobian(q_point).determinant();
    }
    std::cout <<"\n";
    //*/
      
    if (cell->user_pointer() == NULL)
    {
      cell->get_dof_indices (local_dof_indices);
    
      //INTEGRALS FOR BLOCK(0,0) ... matrix A
      for (unsigned int i=0; i < dofs_per_cell; ++i)
        for (unsigned int j=0; j < dofs_per_cell; ++j)
          for (unsigned int q_point=0; q_point < n_q_points; ++q_point)
            cell_matrix(i,j) += ( transmisivity[0] *
                                  fe_values.shape_grad (i, q_point) *
                                  fe_values.shape_grad (j, q_point) *
                                  fe_values.JxW (q_point)
                                );
      //FILLING MATRIX BLOCK A
      for (unsigned int i=0; i < dofs_per_cell; ++i)
        for (unsigned int j=0; j < dofs_per_cell; ++j)
          block_matrix.add( local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i,j)
                            );
    }    
    else
    {
      
      FullMatrix<double>   enrich_cell_matrix;
      std::vector<unsigned int> enrich_dof_indices; //dof indices of enriched and unrenriched dofs
      Vector<double>       enrich_cell_rhs; 
  
  
      Adaptive_integration adaptive_integration(cell,fe,fe_values.get_mapping());
      
      //DBGMSG("cell: %d .................callling adaptive_integration.........",cell->index());
      unsigned int refinement_level = 12;
      
      for(unsigned int t=0; t < refinement_level; t++)
      {
        //DBGMSG("refinement level: %d", t);
        if ( ! adaptive_integration.refine_edge())
          break;
        if (t == refinement_level-1)
        {
          // (output_dir, false, true) must be set to unit coordinates and to show on screen 
          adaptive_integration.gnuplot_refinement(output_dir);
        }
      }
      
      switch(enrichment_method_)
      {
        case Enrichment_method::xfem_ramp: 
          adaptive_integration.integrate_xfem(enrich_cell_matrix, enrich_cell_rhs, enrich_dof_indices, transmisivity[0]);
          break;
        case Enrichment_method::xfem_shift:
          adaptive_integration.integrate_xfem_shift(enrich_cell_matrix, enrich_cell_rhs, enrich_dof_indices, transmisivity[0]);
          break;
        case Enrichment_method::sgfem:
          adaptive_integration.integrate_sgfem(enrich_cell_matrix, enrich_cell_rhs, enrich_dof_indices, transmisivity[0]);
      }
//       //printing enriched nodes and dofs
//       DBGMSG("Printing dof_indices:  [");
//       for(unsigned int a=0; a < enrich_dof_indices.size(); a++)
//       {
//           std::cout << std::setw(3) << enrich_dof_indices[a] << "  ";
//       }
//       std::cout << "]" << std::endl;
      
      //FILLING MATRIX BLOCKs
      
      for (unsigned int i=0; i < enrich_dof_indices.size(); ++i)
      {
        
        for (unsigned int j=0; j < enrich_dof_indices.size(); ++j)
        {
          block_matrix.add( enrich_dof_indices[i],
                            enrich_dof_indices[j],
                            enrich_cell_matrix(i,j)
                            );
        }
        
        block_system_rhs( enrich_dof_indices[i]) += enrich_cell_rhs[i];
      }
      
    } //end for(cells)
    
    /* HOMOGENOUS NEUMANN -> = 0
    for (unsigned int i=0; i<dofs_per_cell; ++i)
    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
    {
      cell_rhs(i) += (fe_values.shape_value (i, q_point) *
        0 *
        fe_values.JxW (q_point));
    }
    //*/

    /* HOMOGENOUS NEUMANN -> = 0
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      system_rhs(local_dof_indices[i]) += cell_rhs(i);
    //*/
  }
  
  for (unsigned int w = 0; w < wells.size(); w++)
  {
    //addition to block (2,2) ... matrix E
    block_matrix.block(2,2).add(w,w,wells[w]->perm2aquitard());
    
    //addition to rhs
    block_system_rhs.block(2)(w) = wells[w]->perm2aquitard() * wells[w]->pressure();
  }
    
  DBGMSG("N_active_cells checkout(on triangulation,integrated): %d \t %d\n", triangulation->n_active_cells() ,count);
  
//   DBGMSG("Printing block matrix:\n");
//   //block_matrix.block(0,0).print_formatted(std::cout);
//   std::cout << "\n\n";
//   block_matrix.print_formatted(std::cout);
//   std::cout << "\n\n";
//   block_system_rhs.print(std::cout);
  //block_matrix.block(2,2).print_formatted(std::cout);
  
  assemble_dirichlet();
  
  if(hanging_nodes)
  {
    hanging_node_constraints.condense(block_matrix);
    hanging_node_constraints.condense(block_system_rhs);
  }
    
}

                               
    
void XModel::solve ()
{
  //how to do things for BLOCK objects
  //http://www.dealii.org/archive/dealii/msg02097.html
  
  SolverControl	solver_control(4000, 1e-10);
  PrimitiveVectorMemory<BlockVector<double> > vector_memory;
 
  
  /*
  //EIGENVALUES ESTIMATE
  
  double biggest_eigen = 0, 
         smallest_eigen = 0;
  double range = 1.0;
  BlockVector<double>::iterator iter = block_solution.begin(),
                                end = block_solution.end();
  for(iter; iter < end; ++iter)
  {
    *iter = (rand() / (double)RAND_MAX) * range;
  }
  EigenPower<BlockVector<double> > eigen_power(solver_control,vector_memory);
  eigen_power.solve<BlockSparseMatrix<double> >(biggest_eigen, block_matrix, block_solution);
  std::cout << std::scientific << "Solver: steps: " << solver_control.last_step() << "\t residuum: " << setprecision(16) << solver_control.last_value() << std::endl;
  std::cout << "Biggest EigenValue = " << biggest_eigen << endl;
  
  
  for(iter = block_solution.begin(); iter < end; ++iter)
  {
    *iter = (rand() / (double)RAND_MAX) * range;
  }
  EigenPower<BlockVector<double> > eigen_power2(solver_control,vector_memory,biggest_eigen);
  eigen_power.solve<BlockSparseMatrix<double> >(smallest_eigen, block_matrix, block_solution);
  std::cout << std::scientific << "Solver: steps: " << solver_control.last_step() << "\t residuum: " << setprecision(16) << solver_control.last_value() << std::endl;
  std::cout << "Smallest EigenValue = " << smallest_eigen << endl;
  
  std::cout << "Estimate of condition number = " << biggest_eigen / smallest_eigen << endl;
  
  for(iter = block_solution.begin(); iter < end; ++iter)
  {
    *iter = 0;
  }
  //*/
  
  
  /*
  //SOLVER SELECTOR
  SolverSelector<BlockVector<double> > solver_selector;
  solver_selector.set_control(solver_control);
  solver_selector.select("cg");
  //*/
  
  
  //USING CG, BICG, PreconditionJacobi
  //SolverBicgstab<BlockVector<double> > solver_bicg(solver_control,vector_memory);
  SolverCG<BlockVector<double> > solver_cg(solver_control, vector_memory);
  
  PreconditionJacobi<BlockSparseMatrix<double> > preconditioning;
  preconditioning.initialize(block_matrix, 1.0);

  solver_cg.solve(block_matrix, block_solution, block_system_rhs, preconditioning); //PreconditionIdentity());
  //solver_bicg.solve(block_matrix, block_solution, block_system_rhs, preconditioning); //PreconditionIdentity());
  
  solver_it = solver_control.last_step();
  
  std::cout << std::scientific << "Solver: steps: " << solver_control.last_step() << "\t residuum: " << setprecision(4) << solver_control.last_value() << std::endl;
  //*/
 
      
  
  //NOT for BLOCK things
  // generate a @p PreconditionSelector
  //PreconditionSelector<SparseMatrix<double>, Vector<double> >
  //  preconditioning("jacobi", 1.);
  //preconditioning.use_matrix(block_matrix);

  
  
  /*
  ReductionControl inner_control (4000, 1.e-16, 1.e-2);
  PreconditionJacobi<BlockSparseMatrix<double> > inner_precondition;
  inner_precondition.initialize(block_matrix, 1.0);
  IterativeInverse<BlockVector<double> > precondition;
  precondition.initialize (block_matrix, inner_precondition);
  precondition.solver.select("cg");
  precondition.solver.set_control(inner_control);
  SolverControl outer_control(1000, 1.e-12);
  SolverRichardson<BlockVector<double> > outer_iteration(outer_control);
  outer_iteration.solve (block_matrix, block_solution, block_system_rhs, precondition);
  //*/
  
  /*
  // DIRECT SOLVER
  SparseDirectUMFPACK direct_solver;
  direct_solver.initialize(block_matrix);
  direct_solver.solve(block_system_rhs);
  block_solution = block_system_rhs;
  //*/
  
  
  if(hanging_nodes)
  {
    hanging_node_constraints.distribute(block_solution);
  }
  
  for (unsigned int w=0; w < wells.size(); ++w)
      std::cout << setprecision(4) << "value of H" << w << " = " << block_solution.block(2)[w] << std::endl;
  
  //DBGMSG("Printing solution:\n");
  //block_solution.print(std::cout);;
  //block_solution.block(0).print(std::cout);
  //std::cout << "\n\n";
  //block_solution.block(1).print(std::cout);
  //*/
}


void XModel::output_results (const unsigned int cycle)
{ 
  // MATRIX OUTPUT
  if(matrix_output_)
  {
    std::stringstream matrix_name;
    matrix_name << "matrix_" << cycle;
    write_block_sparse_matrix(block_matrix,matrix_name.str());
  }
  
  // MESH OUTPUT
  std::stringstream filename1; 
  filename1 << output_dir << "xfem_mesh_" << cycle;
  std::ofstream output1 (filename1.str() + ".msh");
  GridOut grid_out;
  grid_out.write_msh<2> (*triangulation, output1);
   

  // dummy solution for displaying mesh in Paraview
  DataOut<2> data_out;
  data_out.attach_dof_handler (*dof_handler);
  Vector<double> dummys_solution(block_solution.block(0).size());
  data_out.add_data_vector (dummys_solution, "xfem_grid");
  data_out.build_patches (0);
  std::ofstream output2 (filename1.str()+".vtk");
  data_out.write_vtu (output2); 

  std::cout << "\nXFEM mesh written in:\t" << filename1.str() << ".msh \n\t\tand " << filename1.str() << ".vtk" << std::endl;
  
  
  
  //computing solution on the computational mesh
  std::vector< Point< 2 > > support_points(dof_handler->n_dofs());
  DoFTools::map_dofs_to_support_points<2>(fe_values.get_mapping(), *dof_handler, support_points);
  //compute_distributed_solution(support_points);
  
  data_out.clear();
  data_out.attach_dof_handler (*dof_handler);
  
  std::cout << "computing solution on computational mesh" << std::endl;
  
  compute_distributed_solution(support_points);
  
  if(out_decomposed)
  {
    data_out.add_data_vector (dist_unenriched, "xfem_unenriched");
    data_out.add_data_vector (dist_enriched, "xfem_enriched"); 
  }
  data_out.add_data_vector (dist_solution, "xfem_solution");
  
  data_out.build_patches ();

  std::stringstream filename;
  filename << output_dir << "xmodel_solution_" << cycle << ".vtk";
   
  std::ofstream output (filename.str());
  data_out.write_vtk (output);
}


/*                              
void XModel::run (const unsigned int cycle)
{
  if(cycle == 0)
    make_grid();
  else if (is_adaptive)
    refine_grid();
  std::cout << "Number of active cells:       "
            << triangulation->n_active_cells()
            << std::endl;
  std::cout << "Total number of cells: "
            << triangulation->n_cells()
            << std::endl;
  
  

  clock_t start, stop;
  double t = 0.0;

  // Start timer 
  MASSERT((start = clock())!=-1, "Measure time error.");


  if (triangulation_changed == true)
    setup_system();
  assemble_system();

  solve();
 
  // Stop timer 
  stop = clock();
  t = (double) (stop-start)/CLOCKS_PER_SEC;
  printf("Run time: %f\n", t);
}
*/



void XModel::find_dofs_enriched_cells(std::vector<DoFHandler<2>::active_cell_iterator> &cells, const unsigned int &dof_index)
{
  cells.clear();
  for(unsigned int i=0; i < xdata.size(); i++)
  {
    for(unsigned int w=0; w < xdata[i]->n_wells(); w++)
      for(unsigned int k=0; k < xdata[i]->global_enriched_dofs(w).size(); k++)
      {
        if(xdata[i]->global_enriched_dofs(w)[k] == dof_index)
          cells.push_back(xdata[i]->get_cell());
      }
  }
}

void XModel::get_dof_func(const std::vector< Point< 2 > >& points, 
                          const unsigned int& dof_index,
                          dealii::Vector< double >& dof_func, 
                          bool xfem)
{
  MASSERT(points.size() == dof_func.size(), "Vector of solution and support points must be of the same size.");
  if(xfem)
  {
    MASSERT(dof_handler->n_dofs() <= dof_index && dof_index < n_enriched_dofs+dof_handler->n_dofs(), 
            "xfem is true. Given dof index is not index of enriched dof.");
  }
  else
  {
    MASSERT(dof_index < dof_handler->n_dofs(), 
            "xfem is false. Given dof index is not index of unenriched dof.");
  }
    
  std::vector<DoFHandler<2>::active_cell_iterator> cells(GeometryInfo<2>::faces_per_cell);  
  // finds all cells (maximum 4) that has given dof at its node
  find_dofs_enriched_cells(cells, dof_index);
  
  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  //std::pair<DoFHandler<2>::active_cell_iterator, Point<2> > cell_and_point;
  XDataCell *cell_xdata;
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  
  Point<2> unit_point;
  
  for (unsigned int p = 0; p < points.size(); p++)
  {
    dof_func[p] = 0;
    
    //unsigned int c = 0;
    for(unsigned int c = 0; c < cells.size(); c++)
    {
      if(cells[c]->point_inside(points[p]))
      { 
        fe_values.reinit (cells[c]);
        unit_point = fe_values.get_mapping().transform_real_to_unit_cell(cells[c],points[p]);
        //cells[c]->get_dof_indices(local_dof_indices);
        
        if(xfem)
        {
          MASSERT(cells[c]->user_pointer() != NULL,"Cell not enriched!");
        
          cell_xdata = static_cast<XDataCell*>( cells[c]->user_pointer() );
      
          for(unsigned int w = 0; w < cell_xdata->n_wells(); w++)
          {
            for(unsigned int k = 0; k < cell_xdata->global_enriched_dofs(w).size(); k++)
            {
              if(cell_xdata->global_enriched_dofs(w)[k] == dof_index)        
              {
                //for(unsigned int l = 0; l < dofs_per_cell; l++)
                {
                  //if(cell_xdata->global_enriched_dofs()[w][l] !=0 ) //blending function
                  {
                  dof_func[p] += // block_solution(dof_index) *
                                 //fe.shape_value(l, unit_point) * 
                                 fe.shape_value(k, unit_point) *
                                 (cell_xdata->get_well(w)->global_enrich_value(points[p]) 
                                  - cell_xdata->get_well(w)->global_enrich_value(cells[c]->vertex(k))
                                );
                  }
                } //for l
                break;
                //w = cell_xdata->n_wells();
                //k = cell_xdata->global_enriched_dofs(w).size();
              } //if
            } //for k
          } //for w
        } //if(xfem)
        else 
        {
          for(unsigned int i=0; i < dofs_per_cell; i++)
          {
            dof_func[p] += // block_solution(dof_index) *
                           fe.shape_value(i, unit_point);
          }
        }
        
        break;  //preventing sum of values of different cells on faces
      }
    }

  } //for p
  
  /*
  //iteration over all points where we compute solution
  for (unsigned int p = 0; p < points.size(); p++)
  {
    dof_func[p] = 0;
    //DBGMSG("point number: %d\n", p);
    //finds cell where points[p] lies and maps that point to unit_point
    //returns pair<cell, unit_point>
    // cell = cell_and_point.first
    // point = cell_and_point.second
    cell_and_point = GridTools::find_active_cell_around_point<2>(fe_values.get_mapping(), dof_handler, points[p]);
    
    unit_point = GeometryInfo<2>::project_to_unit_cell(cell_and_point.second); //due to roundoffs
    
    fe_values.reinit (cell_and_point.first);
    cell_and_point.first->get_dof_indices(local_dof_indices);

    if(!xfem)
    {
      for(unsigned int j=0; j < dofs_per_cell; j++)
      {
        if(dof_index == local_dof_indices[j])
        dof_func[p] += block_solution(local_dof_indices[j]) *
                      fe.shape_value(j, unit_point);
      }
    }

    
    if (cell_and_point.first->user_pointer() != NULL && xfem)
    {
      cell_xdata = static_cast<XData*>( cell_and_point.first->user_pointer() );
      
      for(unsigned int w = 0; w < cell_xdata->wells().size(); w++)
      {
        for(unsigned int k = 0; k < cell_xdata->global_enriched_dofs()[w].size(); k++)
        {
          if(cell_xdata->global_enriched_dofs()[w][k] == dof_index)        
          {
            for(unsigned int l = 0; l < dofs_per_cell; l++)
            {
              dof_func[p] += block_solution(cell_xdata->global_enriched_dofs()[w][k]) *
                             fe.shape_value(l, unit_point) * 
                             fe.shape_value(k, unit_point) *
                             cell_xdata->wells()[w]->global_enrich_value(points[p]);
            } //for l
          } //if
        } //for k
      } //for w
    } //if
  } //for p
  //*/
}

const dealii::Vector< double >& XModel::get_distributed_solution()
{
  MASSERT(dist_solution.size() > 0,"Distributed solution has not been computed yet.");
  return dist_solution; 
}


const dealii::Vector< double >& XModel::get_solution()
{
  MASSERT(dist_solution.size() > 0,"Solution has not been computed yet.");
  return dist_solution; 
}


void XModel::compute_distributed_solution(const std::vector< Point< 2 > >& points)
{
  unsigned int n_points = points.size(),
               n_vertices = GeometryInfo<2>::vertices_per_cell;
  //clearing distributed solution vectors
  dist_unenriched.reinit(0);
  dist_enriched.reinit(0);
  dist_solution.reinit(0);
  
  //setting new size and initialize with zeros
  dist_unenriched.reinit(n_points);
  dist_enriched.reinit(n_points);
  dist_solution.reinit(n_points);
   

  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  std::pair<DoFHandler<2>::active_cell_iterator, Point<2> > cell_and_point;
  XDataCell *cell_xdata;
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  std::vector<double> local_shape_values (dofs_per_cell);
  
  double xshape;
  
  Point<2> unit_point;
  unsigned int n=1,
               n_const = points.size() / 10;            //how often we will write DBGMSG
  //iteration over all points where we compute solution
  for (unsigned int p = 0; p < points.size(); p++)
  {
    //only writing DBGMSG to see activity
    if(p == n*n_const)
    {
      DBGMSG("point: p=%d\n",p);
      n++;
    }
      
    //writing zero just for sure
    dist_unenriched[p] = 0;
    dist_enriched[p] = 0;
    dist_solution[p] = 0;
    
    //DBGMSG("point number: %d\n", p);
    //finds cell where points[p] lies and maps that point to unit_point
    //returns pair<cell, unit_point>
    // cell = cell_and_point.first
    // point = cell_and_point.second
    cell_and_point = GridTools::find_active_cell_around_point<2>(fe_values.get_mapping(), *dof_handler, points[p]);

    unit_point = GeometryInfo<2>::project_to_unit_cell(cell_and_point.second); //recommended due to roundoffs
    
    fe_values.reinit (cell_and_point.first);
    //unit_point = fe_values.get_mapping().transform_real_to_unit_cell(cell_and_point.first,points[p]);
    
    cell_and_point.first->get_dof_indices(local_dof_indices);

    //compute shape values (will be used futher down) and unenriched part
    for(unsigned int j=0; j < dofs_per_cell; j++)
    {
      local_shape_values[j] = fe.shape_value(j, unit_point);
      dist_unenriched[p] += block_solution(local_dof_indices[j]) *
                            local_shape_values[j];
    }

    
    if (cell_and_point.first->user_pointer() != NULL)
    {
      cell_xdata = static_cast<XDataCell*>( cell_and_point.first->user_pointer() );
      
      for(unsigned int w = 0; w < cell_xdata->n_wells(); w++)
      {
        xshape = cell_xdata->get_well(w)->global_enrich_value(points[p]);
        double ramp = 0;        
        double xshape_inter = 0;
        
        switch(enrichment_method_)
        {
          case Enrichment_method::xfem_shift:
            //compute value (weight) of the ramp function
            for(unsigned int l = 0; l < n_vertices; l++)
            {
              ramp += cell_xdata->weights(w)[l] * local_shape_values[l];
            }
            for(unsigned int k = 0; k < n_vertices; k++)
            {
              dist_enriched[p] += block_solution(cell_xdata->global_enriched_dofs(w)[k]) *
                                  ramp *
                                  local_shape_values[k] *
                                  (xshape - cell_xdata->get_well(w)->global_enrich_value(cell_and_point.first->vertex(k))); //shifted                      
            }
            break;
            
          case Enrichment_method::xfem_ramp: 
            //compute value (weight) of the ramp function
            for(unsigned int l = 0; l < n_vertices; l++)
            {
              ramp += cell_xdata->weights(w)[l] * local_shape_values[l];
            }
            for(unsigned int k = 0; k < n_vertices; k++)
            {
              dist_enriched[p] += block_solution(cell_xdata->global_enriched_dofs(w)[k]) *
                                  ramp *
                                  local_shape_values[k] *
                                  xshape;                      
            }
            break;

          case Enrichment_method::sgfem:
            //compute value interpolant
            for(unsigned int l = 0; l < n_vertices; l++) //M_w
            {
              xshape_inter += local_shape_values[l] * cell_xdata->node_enrich_value(w)[l];
            }
            for(unsigned int k = 0; k < n_vertices; k++)
            {
              if(cell_xdata->global_enriched_dofs(w)[k] != 0)
              dist_enriched[p] += block_solution(cell_xdata->global_enriched_dofs(w)[k]) *
                                  local_shape_values[k] *
                                  (xshape - xshape_inter);
            }
            break;
        } //switch
        
      } //for w
    } //if
    
    dist_solution[p] = dist_enriched[p] + dist_unenriched[p];
  } //for p
}

void XModel::output_distributed_solution(const dealii::Triangulation< 2 > &dist_tria, const unsigned int& cycle, const unsigned int& m_aquifer)
{
  // MATRIX OUTPUT
  if(matrix_output_)
  {
    std::stringstream matrix_name;
    matrix_name << "matrix_" << cycle;
    write_block_sparse_matrix(block_matrix,matrix_name.str());
  }
  
  // MESH OUTPUT
  std::stringstream filename1;
  filename1 << output_dir << "xfem_mesh_" << cycle;
  std::ofstream output1 (filename1.str() + ".msh");
  GridOut grid_out;
  grid_out.write_msh<2> (*triangulation, output1);
  
  //output of refinement flags of persistent triangulation
  std::stringstream filename_flags;
  filename_flags << output_dir << "ref_flags_" << cycle << ".ptf";
  output1.close();
  output1.clear();
  output1.open(filename_flags.str());
  triangulation->write_flags(output1);
  
  
  //dummy solution for displaying mesh int Paraview
  DataOut<2> data_out;
  data_out.attach_dof_handler (*dof_handler);
  Vector<double> dummys_solution(block_solution.block(0).size());
  data_out.add_data_vector (dummys_solution, "xfem_grid");
  data_out.build_patches (0);
  std::ofstream output2 (filename1.str() + ".vtk");
  data_out.write_vtu (output2); 
  std::cout << "\nXFEM mesh written in:\t" << filename1.str() << ".msh \n\t\tand " << filename1.str() << ".vtk" << std::endl;
  data_out.clear();

  
  QGauss<2>        dist_quadrature(2);
  FE_Q<2>          dist_fe(1);                    
  DoFHandler<2>    dist_dof_handler;
  FEValues<2>      dist_fe_values(dist_fe, dist_quadrature, update_default);
  ConstraintMatrix dist_hanging_node_constraints;

  //====================distributing dofs
  dist_dof_handler.initialize(dist_tria,dist_fe);
  
  DoFTools::make_hanging_node_constraints (dist_dof_handler, dist_hanging_node_constraints);  
  dist_hanging_node_constraints.close();

  //====================computing solution on the triangulation
  std::vector< Point< 2 > > support_points(dist_dof_handler.n_dofs());
  
  DoFTools::map_dofs_to_support_points<2>(dist_fe_values.get_mapping(), dist_dof_handler, support_points);
  std::cout << "...computing solution on:   " << mesh_file << std::endl;
  std::cout << "...number of nodes in the mesh:   " << dist_dof_handler.n_dofs() << std::endl;
  std::cout << "...number of nodes in the xfem mesh:   " << dof_handler->n_dofs() << std::endl;
  std::cout << "...number of dofs in the xfem mesh:   " << dof_handler->n_dofs() << " unenriched and " 
            << n_enriched_dofs << " enriched" << std::endl;

  //====================vtk output
  //DataOut<2> data_out;
  data_out.attach_dof_handler (dist_dof_handler);
  
  //Vector<double> dist_solution(dist_dof_handler->n_dofs());
  //get_solution_at_points(support_points, dist_solution);
  //data_out.add_data_vector (dist_solution, "solution");
  
  std::cout << "computing solution on given mesh" << std::endl;
  
  compute_distributed_solution(support_points);
  
  dist_hanging_node_constraints.distribute(dist_unenriched);
  dist_hanging_node_constraints.distribute(dist_enriched);
  dist_hanging_node_constraints.distribute(dist_solution);
  
  if(out_decomposed)
  {
    data_out.add_data_vector (dist_unenriched, "xfem_unenriched");
    data_out.add_data_vector (dist_enriched, "xfem_enriched"); 
  }
  data_out.add_data_vector (dist_solution, "xfem_solution");

  
  data_out.build_patches ();

  std::stringstream filename;
  filename << output_dir << "xmodel_dist_solution_" << cycle << ".vtk";
   
  std::ofstream output (filename.str());
  data_out.write_vtk (output);
  data_out.clear();
  
  std::cout << "\noutput written in:\t" << filename.str() << std::endl;
  
  
  if(out_shape_functions)
  {
    unsigned int n_dofs = dof_handler->n_dofs();
    data_out.attach_dof_handler (dist_dof_handler);
    std::vector<Vector<double> > dist_dof_func;
    //writing only half of the enriched functions
    for(unsigned int i = n_dofs; i < n_enriched_dofs/2+n_dofs; i++)
    {
      dist_dof_func.push_back(Vector<double>(dist_dof_handler.n_dofs()));
      get_dof_func(support_points,i, dist_dof_func.back() );
      DBGMSG("output of xfem shape function of enriched dof %d\n",i);
    }
  
    for(unsigned int i = 0; i < dist_dof_func.size(); i++)
    {
      std::stringstream func_name;
      func_name << "func_" << i+n_dofs;
      data_out.add_data_vector (dist_dof_func[i], func_name.str());
    }
    
    data_out.build_patches ();
  
    std::stringstream filename_x;
    filename_x << output_dir << "xshape_func" << ".vtk";
   
    std::ofstream output_x (filename_x.str());
    data_out.write_vtk (output_x);
  }
  
  //clearing data, releasing pointers (expecially to DofHandler)
  data_out.clear();
}


void XModel::output_distributed_solution(const std::string& mesh_file, const std::string &flag_file, bool is_circle, const unsigned int& cycle, const unsigned int &m_aquifer)
{
  // MATRIX OUTPUT
  if(matrix_output_)
  {
    std::stringstream matrix_name;
    matrix_name << "matrix_" << cycle;
    write_block_sparse_matrix(block_matrix,matrix_name.str());
  }
  
  //triangulation for distributing solution onto domain
  Triangulation<2> dist_coarse_tria;  
  PersistentTriangulation<2> *dist_tria;  
  
  //====================opening mesh
  //open filestream with mesh from GMSH
  std::ifstream in;
  GridIn<2> gridin;
  in.open(mesh_file);
  //attaching the read grid
  gridin.attach_triangulation(dist_coarse_tria);
  if(in.is_open())
  {
    //reading data from filestream
    gridin.read_msh(in);
  }          
  else
  {
    xprintf(Err, "Could not open grid file: %s", mesh_file.c_str());
  }
  
  //creating persistent triangulation
  dist_tria = new PersistentTriangulation<2>(dist_coarse_tria);
  
  //now we will try to load refine flags
  in.close();
  in.clear();
  in.open(flag_file);
  if(in.is_open())
  {
    //if flags file can be read
    dist_tria->read_flags(in);
    if(is_circle)
    {
      DBGMSG("Warning: Be sure that the area of the model is set - the circle triangulation will be computed of it.");
      Point<2> center((down_left+up_right)/2);
      double radius =  down_left.distance(up_right) / 2;
      static const HyperBallBoundary<2> boundary_load(center,radius);
      dist_coarse_tria.set_boundary(0, boundary_load);
    }
  }          
  else if (flag_file != "")
  {
    xprintf(Warn, "Could not open refinement flags file: %s\n Ingore this if loading mesh without refinement flag file.", flag_file.c_str());
  }
  
  //restore in both cases (flags or not)
  dist_tria->restore();
  
  output_distributed_solution(*dist_tria, cycle, m_aquifer);
  
  //destroy persistent triangulation, release pointer to coarse triangulation
  delete dist_tria;
}









































