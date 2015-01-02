
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

//input/output of grid
#include <deal.II/grid/grid_in.h> 

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>


#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <cmath>

#include "parameters.hh"
#include "comparing.hh"
#include "system.hh"
#include "well.hh"

std::vector<Point<2> > Comparing::get_all_quad_points(const std::string mesh_file)
{
  //triangulation for distributing solution onto domain
  Triangulation<2> dist_tria;
  QGauss<2>        dist_quadrature(2);
  FE_Q<2>          dist_fe(1);                    
  DoFHandler<2>    dist_dof_handler(dist_tria);

  //====================opening mesh
  //open filestream with mesh from GMSH
  std::ifstream in;
  GridIn<2> gridin;
  
  in.open(mesh_file);
  //attaching object of triangulation
  gridin.attach_triangulation(dist_tria);
  //reading data from filestream
  gridin.read_msh(in);
  
  //====================distributing dofs
  dist_dof_handler.distribute_dofs(dist_fe);
  FEValues<2>  dist_fe_values(dist_fe, dist_quadrature,update_quadrature_points);
  
  const unsigned int n_q_points = dist_fe_values.n_quadrature_points;
  
  //std::vector<unsigned int> local_dof_indices(dist_fe.dofs_per_cell);
   
  //std::vector<Point<2> > support_points(dist_dof_handler.n_dofs());
  //DoFTools::map_dofs_to_support_points<1,2>( dist_fe_values.get_mapping(), dist_dof_handler, support_points);

  std::cout << "number of cells:" << dist_tria.n_active_cells() << std::endl;
  std::cout << "number of q_points:" << n_q_points << std::endl;
  std::cout << "number of all q_points:" << dist_tria.n_active_cells()*n_q_points << std::endl;
  
  std::vector<Point<2> > points;
  points.reserve(dist_tria.n_active_cells()*n_q_points);
  
  typename DoFHandler<2>::active_cell_iterator
      cell = dist_dof_handler.begin_active(),
      endc = dist_dof_handler.end();
  
  //iteration over cells - on each cell computing addition to every point
  for (; cell != endc; ++cell)
  {
    dist_fe_values.reinit(cell);
  
    const std::vector<Point<2> > &q_points = dist_fe_values.get_quadrature_points();
         
    //iteration over quadrature points (integrating)
    for (unsigned int q=0; q<n_q_points; ++q)
      {
        double r = Parameters::radius;
        Point<2> center1(Parameters::x_dec, 0.0);
        Point<2> center2((-1)*Parameters::x_dec, 0.0);
        
        //checking if the point lies outside the well
        if((q_points[q].distance(center1) > r) && 
            (q_points[q].distance(center2) > r))
        {
          //adding quadrature points
          points.push_back(q_points[q]);
        }
      }
  }
  return points;
}

double Comparing::L2_norm_diff(const dealii::Vector< double >& v1, 
                               const dealii::Vector< double >& v2,
                               const Triangulation< 2 > &tria
                              )
{
  unsigned int n1 = v1.size(),
               n2 = v2.size();
  //DBGMSG("Vector sizes: n1=%d \t n2=%d",n1,n2);
  MASSERT(n1 == n2, "Vectors are not of the same size!");
  
  FE_Q<2> fe(1);
  DoFHandler<2> dof_handler;
  QGauss<2> quad(fe.degree + 2);
  FEValues<2> fe_values(fe,quad, update_values | update_JxW_values);
  
  dof_handler.initialize(tria,fe);
  
  Vector<double> difference(dof_handler.n_dofs());
  std::vector<unsigned int> dof_indices(fe.dofs_per_cell);
  Vector<double> local_diff(fe.dofs_per_cell);
  
  double result=0;
  
  //iterator over cells
  DoFHandler<2>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
      
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(dof_indices);
    
    //local_diff.reinit(fe.dofs_per_cell);
    for(unsigned int q=0; q < quad.size(); q++)
      for(unsigned int i=0; i < fe.dofs_per_cell; i++)
      {
        result += std::pow( (v1[dof_indices[i]] - v2[dof_indices[i]])*fe_values.shape_value(i,q),2) * fe_values.JxW(q);
        //result += std::abs( (v1[dof_indices[i]] - v2[dof_indices[i]])*fe_values.shape_value(i,q)) * fe_values.JxW(q);
      } 
  }
  
  return std::sqrt(result);
}


double Comparing::L2_norm_diff(const dealii::Vector< double >& input_vector, 
                               const Triangulation< 2 > &tria,
                               Function<2>* exact_solution
                              )
{
  FE_Q<2> fe(1);
  DoFHandler<2> dof_handler;
  
  dof_handler.initialize(tria,fe);
  QGauss<2> quad(fe.degree + 2);
  
  Vector<double> difference(dof_handler.n_dofs());
  
  //ConstraintMatrix hanging_node_constraints;
  //DoFTools::make_hanging_node_constraints (dof_handler, hanging_node_constraints);  
  //hanging_node_constraints.close();
  
  VectorTools::integrate_difference<2>(dof_handler,input_vector, *exact_solution, difference, quad,VectorTools::NormType::L2_norm);
  
  double result = 0;
  
  result = difference.l2_norm();
  /*
  std::vector<unsigned int> dof_indices(fe.dofs_per_cell);
  //iterator over cells
  DoFHandler<2>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
      
  for (; cell!=endc; ++cell)
  {
    if(cell->center().distance(center) >= r_enr)
    {
      cell->get_dof_indices(dof_indices);
      
      for(unsigned int i=0; i < fe.dofs_per_cell; i++)
      {
        result += std::pow(v1[dof_indices[i]] - v2[dof_indices[i]],2);
      }
    } 
  }
  
  result = std::sqrt(result/4);
  */
  
  return result;
  //hanging_node_constraints.distribute(solution);
  
}

double Comparing::L2_norm_exact(const dealii::Triangulation< 2 >& tria, 
                                Function<2>* exact_solution)
{
  FE_Q<2> fe(1);
  DoFHandler<2> dof_handler;
  
  dof_handler.initialize(tria,fe);
  QGauss<2> quad(fe.degree + 2);
  
  Vector<double> zero_vector(dof_handler.n_dofs());
  Vector<double> difference(dof_handler.n_dofs());
  VectorTools::integrate_difference<2>(dof_handler,zero_vector, *exact_solution, difference, quad,VectorTools::NormType::L2_norm);
  
  return difference.l2_norm();
}

double Comparing::L2_norm(const dealii::Vector< double >& input_vector, const dealii::Triangulation< 2 >& tria)
{
  FE_Q<2> fe(1);
  DoFHandler<2> dof_handler;
  
  dof_handler.initialize(tria,fe);
  QGauss<2> quad(fe.degree + 2);
  
  Vector<double> difference(dof_handler.n_dofs());
  VectorTools::integrate_difference<2>(dof_handler, input_vector, ZeroFunction<2>(), difference, quad,VectorTools::NormType::L2_norm);
  
  return difference.l2_norm();
}


/******************         SOLUTIONS           **************************/

Solution::ExactBase::ExactBase(Well* well, double radius, double p_dirichlet)
  : Function< 2 >(),
    well_(well),
    radius_(radius),
    p_dirichlet_(p_dirichlet),
    m_(0)
{
//   a_ = (well_->pressure() - p_dirichlet) / (std::log(well_->radius() / radius));
//   b_ = p_dirichlet - a_ * std::log(radius);
    a_ = (well_->radius()*well_->perm2aquifer(m_)*(p_dirichlet_-well_->pressure())) 
          / (1.0 - well_->radius()*well_->perm2aquifer(m_)*std::log(well_->radius()/radius_));
    b_ = p_dirichlet - a_ * std::log(radius);
}


double Solution::ExactSolution::value(const dealii::Point< 2 >& p, const unsigned int /*component*/) const
{
  double distance = well_->center().distance(p);
  if(distance > well_->radius())
    return a_ * std::log(distance) + b_;
  else
    return a_ * std::log(well_->radius()) + b_;//well_->pressure();
}

ExactSolution1::ExactSolution1(Well* well, double radius, double k, double amplitude)
    : ExactBase(well, radius, 0), k_(k), amplitude_(amplitude)
{
    a_ = (well_->radius()*well_->perm2aquifer(m_)*
            (
            well_->pressure() - amplitude_*sin(k_*well_->center()[0])
            - amplitude_*k_*k_*sin(k_*well_->center()[0]) / 2.0 / well_->perm2aquifer(m_)
            )
         ) / 
         (well_->radius()*well_->perm2aquifer(m_)*std::log(well_->radius()/radius_) - 1);
    b_ = - a_ * std::log(radius_);
}

double Solution::ExactSolution1::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
  double distance = well_->center().distance(p);
  if(distance > well_->radius())
    return a_ * std::log(distance) + b_ + amplitude_*std::sin(k_*p[0]);
  else
    return a_ * std::log(well_->radius()) + b_ + amplitude_ * std::sin(k_*p[0]);
}

double Solution::Source1::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
  return amplitude_*k_*k_*std::sin(k_*p[0]);
}

double Solution::ExactSolution2::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
  double distance = well_->center().distance(p);
  if(distance >= well_->radius())
    return a_ * std::log(distance) + b_ + std::sin(k_*p[1]);
  else
    return a_ * std::log(well_->radius()) + b_ + std::sin(k_*p[1]);
}

double Solution::Source2::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
  return k_*k_*std::sin(k_*p[1]);
}


