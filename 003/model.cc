

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

//for adaptive meshes - hanging nodes must be taken care of
#include <deal.II/lac/constraint_matrix.h>

//input/output of grid
#include <deal.II/grid/grid_in.h> 
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>

//for adaptive refinement
#include <deal.II/grid/grid_refinement.h>
//for estimating error
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/numerics/vector_tools.h>

//output
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#define _USE_MATH_DEFINES       //we are using M_PI
#include <cmath>

#include "system.hh"
#include "model.hh"
#include "comparing.hh"
#include "well.hh"
#include "data_cell.hh"

Model::Model ():
    ModelBase::ModelBase(),
    //constant
    triangulation(NULL),
    refinement_percentage(0.3),
    coarsing_percentage(0.0),
    //dealii fem
    fe (1),
    quadrature_formula(2)
{
  name_ = "Default_Adaptive_FEM";
  dof_handler = new DoFHandler<2>();
}

Model::Model (const std::string &name,
              const unsigned int &n_aquifers):
    ModelBase::ModelBase(name,n_aquifers),
    //constant
    triangulation(NULL),
    refinement_percentage(0.3),
    coarsing_percentage(0.0),
    //dealii fem
    fe (1),
    quadrature_formula(2)
{
  dof_handler = new DoFHandler<2>();
}

Model::Model (const std::vector<Well*> &wells, 
              const std::string &name,
              const unsigned int &n_aquifers):
    ModelBase::ModelBase(wells,name,n_aquifers),
    //constant
    triangulation(NULL),
    refinement_percentage(0.3),
    coarsing_percentage(0.0),
    //dealii fem
    fe (1),
    dof_handler(NULL),
    quadrature_formula(2)
{
  dof_handler = new DoFHandler<2>();
}

Model::~Model()
{
  if(dof_handler != NULL)
    delete dof_handler;
  
  if(triangulation != NULL)
    delete triangulation;

  for(unsigned int i=0; i < data_cell.size(); i++)
    delete data_cell[i];
}




void Model::make_grid ()
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
      coarse_tria.refine_global (initial_refinement_); 
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
    filename1 << output_dir_ << "coarse_grid.msh";
 
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

void Model::refine_grid ()
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
      if(cell->center().distance(center) > radius/2.0)
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
            
  /*
  //MESH OUTPUT - LAST USED MESH
   std::stringstream filename1;
   filename1 << output_dir_ << "temp_grid.msh";
 
   std::ofstream output (filename1.str());
  
   GridOut grid_out;
   grid_out.write_msh<2> (triangulation, output);
   */
}

	    
void Model::setup_system ()
{
  //before using block_sp_pattern again, block_matrix must be cleared
  //to destroy pointer to block_sp_pattern
  //else the copy constructor will destroy block_sp_pattern
  //and block_matrix would point to nowhere!!
  block_matrix = 0.0;
  block_matrix.clear();
  
  //clearing data on cells
  triangulation->clear_user_data();
  
  for(unsigned int i=0; i < data_cell.size(); i++)
    delete data_cell[i];
  data_cell.clear();
  
  dof_handler->clear();
  //all data are cleared
    
    
  dof_handler->initialize(*triangulation, fe);
  
  
  std::cout << "Number of degrees of freedom: "
	    << dof_handler->n_dofs()
	    << std::endl;

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
  
  //BLOCK SPARSITY PATTERN
  //inicialization of (temporary) BlockCompressedSparsityPattern, BlockVector
  unsigned int n1 = dof_handler->n_dofs(),	//n1-block(0) dofs
			   n2 = (unsigned int)wells.size(); //n2-block(1) dofs
  BlockCompressedSparsityPattern block_c_sparsity(2, 2);
  block_c_sparsity.block(0,0).reinit(n1,n1);
  block_c_sparsity.block(0,1).reinit(n1,n2);
  block_c_sparsity.block(1,0).reinit(n2,n1);
  block_c_sparsity.block(1,1).reinit(n2,n2);
  block_c_sparsity.collect_sizes();
  
  //sets pattern to block(0,0)
  DoFTools::make_sparsity_pattern(*dof_handler, block_c_sparsity.block(0,0));
  
  //find cell on which the wells are
  find_well_cells();
  
  
  DBGMSG("Printing Data_cell: %d cells on wells boundary\n",data_cell.size());
  n_wells_q_points.clear();
  n_wells_q_points.resize(wells.size(),0);
  for(unsigned int i =0; i < data_cell.size(); i++)
  {
    for(unsigned int w=0; w < data_cell[i]->n_wells(); w++)
    {
      n_wells_q_points[data_cell[i]->get_well_index(w)] += data_cell[i]->q_points(w).size();
    }
    /*
    std::cout << "cell_index: " << data_cell[i]->get_cell()->index() 
             << "\twell_index: " << data_cell[i]->get_well_index(0)
             << "\tnumber of q_points: " << data_cell[i]->get_q_points(0).size()
             << std::endl;
             //*/
  }
  
  for(unsigned int w=0; w < wells.size(); w++)
    DBGMSG("Q_points number check out: %d \t %d\n",wells[w]->q_points().size(), n_wells_q_points[w]);
  
  
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  std::vector<unsigned int> local_dof_indices(dofs_per_cell);
  
  //sets pattern of the wells to block(0,1) and (1,0)
  for (unsigned int d = 0; d < data_cell.size(); d++)
  {
    data_cell[d]->get_cell()->get_dof_indices (local_dof_indices);
    
    for(unsigned int w = 0; w < data_cell[d]->n_wells(); w++)
    {
      for (unsigned int i = 0; i < dofs_per_cell; i++)
      {
        block_c_sparsity.block(0,1).add(local_dof_indices[i], data_cell[d]->get_well_index(w));
        block_c_sparsity.block(1,0).add(data_cell[d]->get_well_index(w), local_dof_indices[i]);
      }
    }
  }
  
  //diagonal pattern in the block (1,1)
  for(unsigned int w = 0; w < wells.size(); w++)
    block_c_sparsity.block(1,1).add(w,w);
      
  //condensing hanging nodes
  hanging_node_constraints.condense(block_c_sparsity);
  //hanging_node_constraints.condense(block_c_sparsity.block(0,0));
 
 
  //copy from (temporary) BlockCompressedSparsityPattern to (main) BlockSparsityPattern
  block_sp_pattern.copy_from(block_c_sparsity);
  
  //prints number of nozero elements in block_c_sparsity
  std::cout << "nozero elements in block_sp_pattern: " << block_sp_pattern.n_nonzero_elements() << std::endl;
  
  if(output_options_ & OutputOptions::output_sparsity_pattern)
  {
    //prints whole BlockSparsityPattern
    std::ofstream out1 (output_dir_ + "block_sp_pattern.1");
    block_sp_pattern.print_gnuplot (out1);

    //prints SparsityPattern of the block (0,0)
    std::ofstream out2 (output_dir_ + "00_sp_pattern.1");
    block_sp_pattern.block(0,0).print_gnuplot (out2);
  }
  
  // END BLOCK SPARSITY PATTERN
 
  //reinitialization of block_matrix
  block_matrix.reinit(block_sp_pattern);
  
  //BLOCK VECTOR - SOLUTION, RHS REINITIALIZATION, filling zeros
  //two blocks in vectors
  block_solution.reinit(2);
  block_system_rhs.reinit(2);
  
  //reinitialization of block_solution 
  //(N,fast=false) .. vector is filled with zeros
  block_solution.block(0).reinit(n1);
  block_solution.block(1).reinit(n2);
  
  //sets zeros into the solution vector
  //block_solution.block(0) = 0.0;
  
  //reinitialization of block_system_rhs
  block_system_rhs.block(0).reinit(n1);
  block_system_rhs.block(1).reinit(n2);
  
  block_solution.collect_sizes();
  block_system_rhs.collect_sizes();
  
//   DBGMSG("Printing sparsity pattern: \n");
//   block_sp_pattern.print(std::cout);
//   std::cout << "\n\n";
}

void Model::find_well_cells()
{
  MASSERT(wells.size() > 0, "No wells are defined in vector of wells");
  
  //DBGMSG("Number of wells to find: %d\n", wells.size());
  //iterator over cells
  DoFHandler<2>::active_cell_iterator cell, endc;
  
  //on each cell iteration over all wells
  for (unsigned int w = 0; w < wells.size(); w++)
  {
    cell = dof_handler->begin_active();
    endc = dof_handler->end();
    
    MASSERT(wells[w]->q_points().size() > 0, 
            "Quadrature point on the well have not been computed yet.");
    
    //checking if number of quadrature points on the well boundary is satisfying
    //according to the measure of the smallest cell
    /*
    double min_diameter = GridTools::minimal_cell_diameter<2>(triangulation);
    double circ_lenght = wells[w]->radius()*2*M_PI;
    double q_dist = circ_lenght / (wells[w]->q_points().size() - 1);
    DBGMSG("Minimal diameter = %f   q_distance = %f\n", min_diameter, q_dist);
    //its like the sampling theorem
    if(min_diameter < 10*q_dist)
    {
      unsigned int new_n_qpoints = 12*circ_lenght / min_diameter + 1;
      wells[w]->evaluate_q_points(new_n_qpoints);
      DBGMSG("New number of q_points = %d  new q_distance = %f\n", new_n_qpoints, circ_lenght/new_n_qpoints);
    }
    //*/
    
    //clearing user flags before using them for each well
    //it is needed for the case the wells are to close (or the mesh is coarse) and lies 
    //on two neighboring cells or even in one cell
    triangulation->clear_user_flags();
    
    //finding the cell in which the first q_point lies
    for (; cell!=endc; ++cell)
    {
      if (cell->point_inside(wells[w]->center()))
      {
        //std::cout << "accessing well: " << w << "\tfrom cell: "<< cell->index() << std::endl; 
        add_data_to_cell(cell, wells[w], w);
        break;
      }
    }
  }
}

void Model::add_data_to_cell (const DoFHandler<2>::active_cell_iterator cell, Well *well, unsigned int well_index)
{
  //std::cout << "CALL add_points_to_cell on cell: " << cell->index() << std::endl;
  // if the flag is set = we have been already there, so continue
    if ( cell->user_flag_set() == true)
    {
      ///std::cout << "\thave already been at cell\n";
      return;
    }
    
  //sets user flag for the cell in which we have been
  cell->set_user_flag();
  
 
  //is the whole cell inside the well ?
  unsigned int vertex_in_count = 0;
  for(unsigned int i=0; i < GeometryInfo<2>::vertices_per_cell; i++)
  {
    if(well->points_inside(cell->vertex(i)))
      vertex_in_count++;
  }
  
  if( ! (vertex_in_count == GeometryInfo<2>::vertices_per_cell))
  {
   
    //temporary vector of q_points
    std::vector<const Point<2>* > points;
    //flag for addition cell to the vector in well object
    bool cell_not_added = true;
  
    //checking if the q_points are in the cell
    for(unsigned int p=0; p < well->q_points().size(); p++)
    {
      if (cell->point_inside(well->q_points()[p]))
      {
        cell_not_added = false;
        //adding point
        points.push_back( &(well->q_points()[p]) );
      }
    }
  
    //if cell is added then add points
    if (cell_not_added == false)
    {
      if(cell->user_pointer() == NULL)
      {
        data_cell.push_back(new DataCell(cell, well, well_index, points));    
        cell->set_user_pointer(data_cell.back());
      }
      else
      {
        DataCell* data_pointer = static_cast<DataCell*> (cell->user_pointer());
        data_pointer->add_data(well, well_index, points);
      }
      ///std::cout << "\tadded." << std::endl;
    }

  } //if cell is inside the well then look at the neighbours
  //we will go through all cell inside the well too - then cannot miss any q_point..
  
  // see step30, method assemble_system2()
  // searching neighbors...
  for (unsigned int face_no=0; face_no < GeometryInfo<2>::faces_per_cell; ++face_no)
  {
    ///std::cout << "\tface(" << face_no << "): ";
    typename DoFHandler<2>::face_iterator face = cell->face(face_no);
    
    //if the face is at the boundary, there is no neighbor, so continue
    if (face->at_boundary()) 
    {
      ///std::cout << "at the boundary\n";
      continue;
    }
    
        
    // asking face about children - if so, then  there must also be finer cells 
    // which are children or farther offsprings of our neighbor.
    if (face->has_children())
      {
        ///std::cout << "\t Face has children." << std::endl;
    
        // iteration over subfaces - children
        for (unsigned int subface_no = 0; subface_no < face->number_of_children(); ++subface_no)
          {
            typename DoFHandler<2>::cell_iterator neighbor_child
                           = cell->neighbor_child_on_subface (face_no, subface_no);
            Assert (!neighbor_child->has_children(), ExcInternalError());
            
            ///std::cout << "\tFace=" << face_no << " subface=" << subface_no 
            ///          << " entering cell=" << neighbor_child->index() << std::endl;
            // entering on the neighbor's children behind the current face
            add_data_to_cell(neighbor_child, well, well_index);
          }
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
            ///std::cout << "\tneigh: " << neighbor->index() << "\t same refine level: " 
            ///          << neighbor->level() << std::endl;
            add_data_to_cell(neighbor, well, well_index);
          }
        else
          //is coarser
          {
            ///std::cout << "\tneigh: " << neighbor->index() << "\t is coarser" << std::endl; 
            add_data_to_cell(neighbor, well, well_index);
          }
      }
  }
}


void Model::assemble_system ()
{
  // set update flags
  UpdateFlags update_flags  = update_gradients | update_JxW_values;
  if (rhs_function) update_flags = update_flags | update_values | update_quadrature_points;
  
  FEValues<2> fe_values (fe, quadrature_formula,update_flags);
  
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  
  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);	//HOMOGENOUS NEUMANN -> = 0
  
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  
  
  //cell (of the well) matrix dimensions
  // +1 is for the well boundary equation (blocks C,D)
  const unsigned int wm = dofs_per_cell + 1;
  FullMatrix<double> well_cell_matrix (wm,wm); 
  std::vector<unsigned int> well_local_dof_indices (wm); 
  Vector<double> shape_value(wm);
  
  
  unsigned int count = 0;
  DoFHandler<2>::active_cell_iterator
    cell = dof_handler->begin_active(),
    endc = dof_handler->end();
  for (; cell!=endc; ++cell)
  {
    count++;
    fe_values.reinit (cell);
    cell->get_dof_indices (local_dof_indices);
    
    cell_matrix = 0;
    cell_rhs = 0;		//HOMOGENOUS NEUMANN -> = 0
    
    //INTEGRALS FOR BLOCK(0,0) ... matrix A
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
        //for (unsigned int w = 0; w < wells.size(); w++)
        //{
          //if( ! wells[w]->points_inside(fe_values.get_quadrature_points()[q_point]))
          {
            cell_matrix(i,j) += ( transmisivity[0] *
                                  fe_values.shape_grad (i, q_point) *
                                  fe_values.shape_grad (j, q_point) *
                                  fe_values.JxW (q_point) );
            // break;
          }
        //}
        }
        
    if(rhs_function != nullptr)
    {
      // HOMOGENOUS NEUMANN -> = 0, else source term
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
          cell_rhs(i) += (fe_values.shape_value (i, q_point) *
                          rhs_function->value(fe_values.quadrature_point(q_point)) * // 0 for homohenous neumann
                          fe_values.JxW (q_point));
        } 
      block_system_rhs.add(local_dof_indices, cell_rhs);
    }

    //FILLING MATRIX BLOCK(0,0) ... matrix A
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        block_matrix.block(0,0).add ( local_dof_indices[i],
                                      local_dof_indices[j],
                                      cell_matrix(i,j) );
  
    //WELLS
    if(cell->user_pointer() != NULL)
    {
      //DBGMSG("computing well\n");
      DataCell* data = static_cast<DataCell*> (cell->user_pointer());
      
      for (unsigned int w = 0; w < data->n_wells(); w++)
      {
        //DBGMSG("well number: %d\n",w);
        Well * well = data->get_well(w);
        //jacobian = radius of the well; weights are the same all around
        double jxw = 2 * M_PI * well->radius() / n_wells_q_points[data->get_well_index(w)];

        //reseting the local matrix
        well_cell_matrix = 0;
        shape_value = 0;
        
        well_local_dof_indices.clear();
        well_local_dof_indices = local_dof_indices;
        well_local_dof_indices.push_back(data->get_well_index(w));
        
        well_local_dof_indices[dofs_per_cell]=data->get_well_index(w) + dof_handler->n_dofs();
      
        //reinitialization of fe_values - is done before
        //fe_values.reinit(data->get_cell());
      
        //cycle over quadrature points inside the current cell
        for (unsigned int q=0; q < data->q_points(w).size(); ++q)
        {
          //we are mapping the point only to map the shape function
          //we do not need to use JxW like in integrals !!
          //http://www.dealii.org/developer/doxygen/deal.II/classMapping.html
        
          Point<2> unit_point = fe_values.get_mapping().transform_real_to_unit_cell(data->get_cell(),
                                                      *(data->q_points(w)[q]) );
          
          //getting shape values in vector [fe.shapevalue; -1.0]
          for (unsigned int i=0; i < dofs_per_cell; ++i)
            shape_value[i] = fe.shape_value (i, unit_point);
          shape_value[dofs_per_cell] = -1.0;  // blocks C is negative
            

          for (unsigned int i=0; i < wm; ++i)
            for (unsigned int j=0; j < wm; ++j)
            { 
              well_cell_matrix(i,j) += ( wells[w]->perm2aquifer() *
                                         shape_value[i] *
                                         shape_value[j] *
                                         jxw );
            }
        } //end of iteration over q_points
      
      //well_cell_matrix.print_formatted(std::cout);
      //FILLING BLOCK MATRIX
      for (unsigned int i=0; i < wm; ++i)
        for (unsigned int j=0; j < wm; ++j)
        {
          //DBGMSG("well addition: %d %d -- %d %d\n",i,j,well_local_dof_indices[i],well_local_dof_indices[j]);
          block_matrix.add ( well_local_dof_indices[i],
                             well_local_dof_indices[j],
                             well_cell_matrix(i,j) );
        }     
      }
    } //end if user pointer 
  //*/
  } //end of iteration over cells
    
  //DBGMSG("cell iteration finished.\n");
  for (unsigned int w = 0; w < wells.size(); w++)
  {
    //addition to block (1,1) ... matrix D
    block_matrix.block(1,1).add(w, w, wells[w]->perm2aquitard() );
  
    //addition to rhs
    block_system_rhs.block(1)(w) = wells[w]->perm2aquitard() * wells[w]->pressure();
  } //end of iteration over wells
 

  DBGMSG("N_active_cells checkout(on triangulation,integrated): %d \t %d\n", triangulation->n_active_cells() ,count);
  //block_matrix.block(0,0).print_formatted(std::cout);

  /*
  //setting dofs inside the well
  for (unsigned int w = 0; w < wells.size(); w++)
  {
    //iteration over cells through which the well boundary goes
    for (unsigned int c=0; c < wells[w]->cells.size(); c++)
    {
      for (unsigned int i=0; i < dofs_per_cell; i++)
      {
        if( wells[w]->points_inside(wells[w]->cells[c]->vertex(i)))
        {
          unsigned int dof = wells[w]->cells[c]->vertex_dof_index(i,0);
          for(unsigned int j = 0;  j < (block_matrix.block(0,0).n()+wells.size()); j++)
            block_matrix.set(dof,j,0);
          
          block_matrix.set(dof,dof,1);
          block_system_rhs(dof) = wells[w]->pressure();
        }
      }
    }
  }
  //*/
  
  assemble_dirichlet();
  
  hanging_node_constraints.condense(block_matrix);
  hanging_node_constraints.condense(block_system_rhs);
}
                               
    
void Model::solve ()
{
  //block_matrix.print_formatted(std::cout);
  //block_system_rhs.print(std::cout);
  SolverControl	solver_control(solver_max_iter_, solver_tolerance_);
  PrimitiveVectorMemory<BlockVector<double> > vector_memory;
  //this solver is used for block matrices and vectors
  SolverCG<BlockVector<double> > solver(solver_control, vector_memory);
  
  PreconditionJacobi<BlockSparseMatrix<double> > preconditioner;
  preconditioner.initialize(block_matrix, 1.0);
			  
  solver.solve(block_matrix, block_solution, block_system_rhs, preconditioner); //PreconditionIdentity());
  
  solver_iterations_ = solver_control.last_step();
  std::cout << std::scientific << "Solver: steps: " << solver_control.last_step() << "\t residuum: " << setprecision(4) << solver_control.last_value() << std::endl;
  
  hanging_node_constraints.distribute(block_solution);
  //hanging_node_constraints.distribute(block_solution.block(0));
}


void Model::output_results (const unsigned int cycle)
{
    // MATRIX OUTPUT
    if(output_options_ & OutputOptions::output_matrix)
        write_block_sparse_matrix(block_matrix,"fem_matrix");
  
    //MESH OUTPUT
  
    if(output_options_ & OutputOptions::output_gmsh_mesh)
    {
        std::stringstream filename;
        filename << output_dir_ << "real_grid_" << cycle;
        std::ofstream output (filename.str() + ".msh");
        GridOut grid_out;
        grid_out.write_msh<2> (*triangulation, output);
        
            //output of refinement flags of persistent triangulation
        std::stringstream filename1;
        filename1 << output_dir_ << "ref_flags_" << cycle << ".ptf";
        output.close();
        output.clear();
        output.open(filename1.str());
        //std::ofstream output1 (filename1.str());
        triangulation->write_flags(output);
    }
   

   
   
   //SOLUTION OUTPUT
   DataOut<2> data_out;

   data_out.attach_dof_handler (*dof_handler);
   data_out.add_data_vector (block_solution.block(0), "fem_solution");
   data_out.build_patches ();

   DBGMSG("output_results\n");
   std::stringstream filename2;
   filename2 << output_dir_ << "solution_" << cycle << ".vtk";
  
   std::cout << "output written in: " << filename2.str() << std::endl;
   
   std::ofstream output2 (filename2.str());
   data_out.write_vtk (output2);
   
   for (unsigned int w=0; w < wells.size(); ++w)
      std::cout << "value of H" << w << " = " << block_solution.block(1)[w] << std::endl;
   
   
}


const dealii::Vector< double >& Model::get_distributed_solution()
{
  //MASSERT(0,"Is not implemented in class Model.");
  return dist_solution; 
}


const dealii::Vector< double >& Model::get_solution()
{
  MASSERT(block_solution.block(0).size() > 0,"Solution has not been computed yet.");
  return block_solution.block(0); 
}

void Model::output_distributed_solution(const std::string& mesh_file, const std::string &flag_file, bool is_circle, 
                                        const unsigned int& cycle, const unsigned int& m_aquifer)
{ 
  { MASSERT(0, "Not implemented in class Model.");}
}

void Model::output_distributed_solution(const dealii::Triangulation< 2 >& dist_tria, const unsigned int& cycle)
{
  // MATRIX OUTPUT
  if(output_options_ & OutputOptions::output_matrix)
  {
    std::stringstream matrix_name;
    matrix_name << "matrix_" << cycle;
    write_block_sparse_matrix(block_matrix,matrix_name.str());
  }
  
  
  FE_Q<2>          dist_fe(1);                    
  DoFHandler<2>    dist_dof_handler;
  ConstraintMatrix hanging_node_constraints;
  
  //====================distributing dofs
  dist_dof_handler.initialize(dist_tria,dist_fe);
  
  DoFTools::make_hanging_node_constraints (dist_dof_handler, hanging_node_constraints);  
  hanging_node_constraints.close();
  
  dist_solution.reinit(dist_dof_handler.n_dofs());
  
  VectorTools::interpolate_to_different_mesh(*dof_handler, 
                                             block_solution.block(0), 
                                             dist_dof_handler, 
                                             hanging_node_constraints, 
                                             dist_solution);
  
  DataOut<2> data_out;
  data_out.attach_dof_handler (dist_dof_handler);
  
  data_out.add_data_vector (dist_solution, "fem_solution");
  data_out.build_patches ();

  std::stringstream filename;
  filename  << output_dir_ << "fem_dist_solution_" << cycle << ".vtk";
   
  std::ofstream output (filename.str());
  data_out.write_vtk (output);
  data_out.clear();
  
  std::cout << "\noutput written in:\t" << filename.str() << std::endl;
}








void Model::output_foreign_results(const unsigned int cycle, const Vector<double> &foreign_solution)
{

   DataOut<2> data_out;

   data_out.attach_dof_handler (*dof_handler);
   data_out.add_data_vector (foreign_solution, "solution");
   data_out.build_patches ();

   std::stringstream filename;
   filename << output_dir_ << "solution_foreign_" << cycle << ".vtk";
   
   std::ofstream output (filename.str());
   data_out.write_vtk (output);
   std::cout << "L2_NORM_bem:   " << foreign_solution.l2_norm() << std::endl;
}




std::pair< double, double > Model::integrate_difference(dealii::Vector< double >& diff_vector, 
                                                        const Function< 2 >& exact_solution)
{
    std::pair<double,double> norms;

    std::cout << "Computing l2 norm of difference...";
    unsigned int dofs_per_cell = fe.dofs_per_cell,
                 index = 0;
                 
    double exact_value, value, cell_norm, total_norm, nodal_norm, total_nodal_norm;
    double distance_treshold = 5.0;
             
    QGauss<2> temp_quad(3);
    FEValues<2> temp_fe_values(fe,temp_quad, update_values | update_quadrature_points | update_JxW_values);
    std::vector<unsigned int> local_dof_indices (temp_fe_values.dofs_per_cell);   
  
    //Check if the dofs of FE_DGQ are corresponding.
//         FE_DGQ<2> temp_fe(0);
//         DoFHandler<2>    temp_dof_handler;
//         ConstraintMatrix hanging_node_constraints;
//         temp_dof_handler.initialize(*triangulation,temp_fe);
//         DoFTools::make_hanging_node_constraints (temp_dof_handler, hanging_node_constraints);  
//         hanging_node_constraints.close();
//         Vector<double> my_vector(dof_handler->get_tria().n_active_cells());
//         DoFHandler<2>::active_cell_iterator my_cell = temp_dof_handler.begin_active();
//         std::vector<unsigned int> my_local_dof_indices (temp_fe.dofs_per_cell);  
        
    Vector<double> diff_nodal_vector(dof_handler->n_dofs());
    diff_vector.reinit(dof_handler->get_tria().n_active_cells());
    
    DoFHandler<2>::active_cell_iterator
        cell = dof_handler->begin_active(),
        endc = dof_handler->end();
    for (; cell!=endc; ++cell)
    {
        cell_norm = 0;
        //DBGMSG("cell: %d\n",cell->index());
        // is there is NOT a user pointer on the cell == is not enriched?
        temp_fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        
//          if (cell->user_pointer() == nullptr)
        
//         if(cell->center().distance(wells[0]->center()) 
//             > (wells[0]->radius() + cell->diameter()/2) + distance_treshold)
         {
            for(unsigned int q=0; q < temp_fe_values.n_quadrature_points; q++)
            {
                value = 0;
                for(unsigned int i=0; i < dofs_per_cell; i++)
                    value += block_solution(local_dof_indices[i]) * temp_fe_values.shape_value(i,q);
                
                exact_value = exact_solution.value(temp_fe_values.quadrature_point(q));
                value = value - exact_value;                         // u_h - u
                cell_norm += value * value * temp_fe_values.JxW(q);  // (u_h-u)^2 * JxW
            }
        }
        //TODO: use also adaptive integration
//         else
//         { 
//             Adaptive_integration adaptive_integration(cell, fe, temp_fe_values.get_mapping());
//             
//             //unsigned int refinement_level = 15;
//             for(unsigned int t=0; t < adaptive_integration_refinement_level_; t++)
//             {
//                 //if(t>0) DBGMSG("refinement level: %d\n", t);
//                 if ( ! adaptive_integration.refine_edge())
//                 break;
//                 if (t == adaptive_integration_refinement_level_-1)
//                 {
//                     // (output_dir, false, true) must be set to unit coordinates and to show on screen 
//                     //adaptive_integration.gnuplot_refinement(output_dir_);
//                 }
//             }
//             cell_norm = adaptive_integration.integrate_l2_diff<EnrType>(block_solution,exact_solution);
//         }
        
        cell_norm = std::sqrt(cell_norm);// / cell->measure());   // square root
        diff_vector[index] = cell_norm;     // save L2 norm on cell
        
    //Check if the dofs of FE_DGQ are corresponding.
//         my_cell->get_dof_indices(my_local_dof_indices);
//         my_vector[index] = my_local_dof_indices[0];
//         ++my_cell;
        
        index ++;
        
        //node values should be exactly equal FEM dofs
        for(unsigned int i=0; i < dofs_per_cell; i++)
        {
            nodal_norm = block_solution(local_dof_indices[i]) - exact_solution.value(cell->vertex(i));
            diff_nodal_vector[local_dof_indices[i]] = std::abs(nodal_norm);
        }
    }
    
    total_nodal_norm = diff_nodal_vector.l2_norm();
    total_norm = diff_vector.l2_norm();
    std::cout << "\t" << total_norm << "\t vertex l2 norm: " << total_nodal_norm << std::endl;
    
    if(output_options_ & OutputOptions::output_error)
    {
        FE_DGQ<2> temp_fe(0);
        DoFHandler<2>    temp_dof_handler;
        ConstraintMatrix hanging_node_constraints;
  
        temp_dof_handler.initialize(*triangulation,temp_fe);
  
        DoFTools::make_hanging_node_constraints (temp_dof_handler, hanging_node_constraints);  
        hanging_node_constraints.close();
  
        //====================vtk output
        DataOut<2> data_out;
        data_out.attach_dof_handler (temp_dof_handler);
  
        hanging_node_constraints.distribute(diff_vector);
  
        data_out.add_data_vector (diff_vector, "fem_error");
        //Check if the dofs of FE_DGQ are corresponding.
        //data_out.add_data_vector (my_vector, "my_vector");
        data_out.build_patches ();

        std::stringstream filename;
        filename << output_dir_ << "model_error_" << cycle_ << ".vtk";
   
        std::ofstream output (filename.str());
        if(output.is_open())
        {
            data_out.write_vtk (output);
            data_out.clear();
            std::cout << "\noutput(error) written in:\t" << filename.str() << std::endl;
        }
        else
        {
            std::cout << "Could not write the output in file: " << filename.str() << std::endl;
        }
    }
    
    return std::make_pair(total_nodal_norm, total_norm);
}









/** ******************************************************************
 *                                OBSOLETE        
 * *******************************************************************
 */

///OBSOLETE - finding cells in which the centers of the wells lie.
/*
void Model::FindWellCells(FEValues<2> *fe_values)
{
  //iterator over cells
  DoFHandler<2>::active_cell_iterator
    cell = dof_handler->begin_active(),
    endc = dof_handler->end();
  //point in the unit cell
  Point<2> point;
  
  //count exceptions in mapping
  unsigned int mapp_exc_counter = 0;
  for (; cell!=endc; ++cell)
    {
    //on each cell iteration over all wells
      for (unsigned int w = 0; w < wells.size(); w++)
      {
        try
        {
          //point in the unit cell
          point = fe_values->get_mapping().transform_real_to_unit_cell(cell,wells[w]->GetPosition());
          //checking if the point is in the unit cell
          if( point(0) >= 0 && point(0) <= 1 &&
              point(1) >= 0 && point(1) <= 1)
            {
              //setting size of the vector with dof indices
              wells[w]->well_dof_indices.resize(fe.dofs_per_cell);
              //writing dof indices in the vector
              cell->get_dof_indices(wells[w]->well_dof_indices);
              //writing the cell iterator
              wells[w]->cell = cell;
            }
        }
        catch(...)
        {
          mapp_exc_counter++;
        }
      }
    }
  
  std::cout << "mapping exceptions in FindWellCells" << 
               " means that well does not lie in the cell: " << 
               mapp_exc_counter << std::endl;
}
//*/

