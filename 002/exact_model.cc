#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/dofs/dof_tools.h>

//input/output of grid
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h> 
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/persistent_tria.h>


//for adaptive refinement
#include <deal.II/grid/grid_refinement.h>

//output
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#include "system.hh"
#include "exact_model.hh"
#include "well.hh"

ExactModel::ExactModel()
{

}

ExactModel::ExactModel(ExactBase* exact_solution)
: exact_solution(exact_solution),
  dist_tria(NULL)
{}

ExactModel::~ExactModel()
{
  if(dist_tria != NULL)
    delete dist_tria;
}


void ExactModel::output_distributed_solution(const std::string &flag_file,
                                             const unsigned int &cycle)
{
  if(dist_tria != NULL)
  {
    delete dist_tria;
    dist_coarse_tria.clear();
  }
  
  GridGenerator::hyper_ball<2>(dist_coarse_tria,exact_solution->well()->center(),exact_solution->radius());
  static const HyperBallBoundary<2> boundary(exact_solution->well()->center(),exact_solution->radius());
  dist_coarse_tria.set_boundary(0, boundary);
      
  dist_tria = new PersistentTriangulation<2>(dist_coarse_tria);
  std::ifstream in;
  in.open(flag_file);
  if(in.is_open())
    dist_tria->read_flags(in);
  else
  {
    xprintf(Err, "Could not open refinement flags file: %s\n", flag_file.c_str());
  }
  //creates actual grid to be available
  dist_tria->restore();
  
  output_distributed_solution(*dist_tria, cycle);
  
  //destroy persistent triangulation, release pointer to coarse triangulation
  //delete dist_tria;
}



void ExactModel::output_distributed_solution(const dealii::Triangulation< 2 >& dist_tria, const unsigned int& cycle)
{
  QGauss<2>        dist_quadrature(2);
  FE_Q<2>          dist_fe(1);                    
  DoFHandler<2>    dist_dof_handler;
  FEValues<2>      dist_fe_values(dist_fe, dist_quadrature, update_default);
  ConstraintMatrix hanging_node_constraints;
  
  //====================distributing dofs
  dist_dof_handler.initialize(dist_tria,dist_fe);
  
  DoFTools::make_hanging_node_constraints (dist_dof_handler, hanging_node_constraints);  
  hanging_node_constraints.close();
  
  //====================computing solution on the triangulation
  std::vector< Point< 2 > > support_points(dist_dof_handler.n_dofs());
  solution.reinit(dist_dof_handler.n_dofs());
  
  DoFTools::map_dofs_to_support_points<2>(dist_fe_values.get_mapping(), dist_dof_handler, support_points);
  std::cout << "number of nodes in the mesh:   " << dist_dof_handler.n_dofs() << std::endl;

  //====================vtk output
  DataOut<2> data_out;
  data_out.attach_dof_handler (dist_dof_handler);
  
  for(unsigned int p=0; p < support_points.size(); p++)
  {
    solution[p] = exact_solution->value(support_points[p]);
  }
  
  hanging_node_constraints.distribute(solution);
  
  data_out.add_data_vector (solution, "exact_solution");
  data_out.build_patches ();

  std::stringstream filename;
  filename  << "../output/exact_solution_" << cycle << ".vtk";
   
  std::ofstream output (filename.str());
  if(output.is_open())
  {
    data_out.write_vtk (output);
    data_out.clear();
    std::cout << "\noutput written in:\t" << filename.str() << std::endl;
  }
  else
  {
    std::cout << "Could not write the output in file: " << filename.str() << std::endl;
  }
  
}

