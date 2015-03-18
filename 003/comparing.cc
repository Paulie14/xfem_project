
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>


#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <cmath>

#include "comparing.hh"
#include "system.hh"
#include "well.hh"

using namespace dealii;
using namespace compare;

namespace compare {
    
double L2_norm_diff(const Vector< double >& v1, 
                    const Vector< double >& v2,
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


double L2_norm_diff(const Vector< double >& input_vector, 
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

double L2_norm_exact(const Triangulation< 2 >& tria, 
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

double L2_norm(const Vector< double >& input_vector, 
               const Triangulation< 2 >& tria)
{
  FE_Q<2> fe(1);
  DoFHandler<2> dof_handler;
  
  dof_handler.initialize(tria,fe);
  QGauss<2> quad(fe.degree + 2);
  
  Vector<double> difference(dof_handler.n_dofs());
  VectorTools::integrate_difference<2>(dof_handler, input_vector, ZeroFunction<2>(), difference, quad,VectorTools::NormType::L2_norm);
  
  return difference.l2_norm();
}
} // compare

/******************         SOLUTIONS           **************************/

ExactBase::ExactBase(Well* well, double radius, double p_dirichlet)
  : Function< 2 >(),
    well_(well),
    radius_(radius),
    p_dirichlet_(p_dirichlet),
    m_(0)
{
//   a_ = (well_->pressure() - p_dirichlet) / (std::log(well_->radius() / radius));
//   b_ = p_dirichlet - a_ * std::log(radius);
    if(well->is_active())
    {
        a_ = (well_->radius()*well_->perm2aquifer(m_)*(p_dirichlet_-well_->pressure())) 
            / (1.0 - well_->radius()*well_->perm2aquifer(m_)*std::log(well_->radius()/radius_));
        b_ = p_dirichlet - a_ * std::log(radius);
    }
    else
    {
        a_ = 0;
        b_ = 0;
    }
}


double ExactSolutionZero::value(const Point< 2 >& p, const unsigned int component) const
{
    return 0;
}

Tensor< 1, 2 > ExactSolutionZero::grad(const Point< 2 >& p, const unsigned int component) const
{
    Tensor<1,2> grad;
    return grad;
}




double ExactSolution::value(const dealii::Point< 2 >& p, const unsigned int /*component*/) const
{
  double distance = well_->center().distance(p);
  if(distance > well_->radius())
    return a_ * std::log(distance) + b_;
  else
    return a_ * std::log(well_->radius()) + b_;//well_->pressure();
}

Tensor< 1, 2 > ExactSolution::grad(const Point< 2 >& p, const unsigned int component) const
{
    double distance = well_->center().distance(p);
    Tensor<1,2> grad;
    if(distance > well_->radius())
    {
        grad = p;
        return a_ * 2.0/(distance*distance)*grad;
    }
    else
        return grad;
}


ExactSolution1::ExactSolution1(Well* well, double radius, double k, double amplitude)
    : ExactBase(well, radius, 0), k_(k), amplitude_(amplitude)
{
    double delta = std::log(well_->radius()/radius_);
    double gamma = 1.0 / (1.0 - well_->radius()*well->perm2aquifer(m_)*delta);
    double temp = well_->perm2aquitard(m_)/well_->perm2aquifer(m_);
    double well_pressure = temp*well_->pressure() / (gamma + temp);
    
//     if(well->is_active())
//     {
//     a_ = (well_->radius()*well_->perm2aquifer(m_)*
//             (
//             well_pressure - amplitude_*sin(k_*well_->center()[0])
//             - amplitude_*k_*k_*sin(k_*well_->center()[0]) / 2.0 / well_->perm2aquifer(m_)
//             )
//          ) / 
//          (well_->radius()*well_->perm2aquifer(m_)*delta - 1);
//     b_ = - a_ * std::log(radius_);
//     }
    double twopirosigma = 2*M_PI*well_->radius()*well_->perm2aquifer(m_);
    if(well->is_active())
    {
    a_ = (twopirosigma * (well_pressure - amplitude_*std::sin(k_*well_->center()[0]))
           - 0.5*amplitude_*k_*k_*std::sin(k_*well_->center()[0]) )
          /
          (-1.0/well_->radius() + twopirosigma*delta);
    b_ = - a_ * std::log(radius_);
    }
    else
    {
        a_ = 0;
        b_ = 0;
    }
}

double ExactSolution1::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
  double distance = well_->center().distance(p);
  if(distance > well_->radius())
    return a_ * std::log(distance) + b_ + amplitude_*std::sin(k_*p[0]);
  else
    return a_ * std::log(well_->radius()) + b_ + amplitude_ * std::sin(k_*p[0]);
}

Tensor< 1, 2 > ExactSolution1::grad(const Point< 2 >& p, const unsigned int component) const
{
    MASSERT(0,"Not implemented!!!");
    Tensor<1,2> grad;
    return grad;
}


double Source1::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
  return amplitude_*k_*k_*std::sin(k_*p[0]);
}

double ExactSolution2::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
  double distance = well_->center().distance(p);
  if(distance >= well_->radius())
    return a_ * std::log(distance) + b_ + std::sin(k_*p[1]);
  else
    return a_ * std::log(well_->radius()) + b_ + std::sin(k_*p[1]);
}

Tensor< 1, 2 > ExactSolution2::grad(const Point< 2 >& p, const unsigned int component) const
{
    MASSERT(0,"Not implemented!!!");
    Tensor<1,2> grad;
    return grad;
}


double Source2::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
  return k_*k_*std::sin(k_*p[1]);
}







ExactSolution3::ExactSolution3(Well* well, double radius, double k, double amplitude)
    : ExactBase(well, radius, 0), k_(k), amplitude_(amplitude)
{
    if(well->is_active())
    {
        a_ = (well_->radius()*well_->perm2aquifer(m_)*(p_dirichlet_-well_->pressure())) 
            / (1.0 - well_->radius()*well_->perm2aquifer(m_)*std::log(well_->radius()/radius_));
        b_ = p_dirichlet_ - a_ * std::log(radius_);
    }
    else
    {
        a_ = 0;
        b_ = 0;
    }
}

Tensor< 1, 2 > ExactSolution3::grad(const Point< 2 >& p, const unsigned int component) const
{
    double distance = well_->center().distance(p);
    Tensor<1,2> grad;
    if(distance > well_->radius())
    {
        grad[0]= (p[0] - well_->center()[0]);
        grad[1]= (p[1] - well_->center()[1]);
        grad = grad * 2*(distance - well_->radius())/distance * std::sin(k_*p[0]);
        grad[0] += (distance-well_->radius())*(distance-well_->radius())*k_*std::cos(k_*p[0]);
        return amplitude_ * grad;
    }
    else
        return grad;
}


double ExactSolution3::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
  double distance = well_->center().distance(p);
  double distance_from_well = distance - well_->radius();
  if(distance > well_->radius())
    return a_ * std::log(distance) + b_ + distance_from_well*distance_from_well * amplitude_*std::sin(k_*p[0]);
  else
    return a_ * std::log(well_->radius()) + b_;
}

double Source3::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
    double distance = well_->center().distance(p);
    double distance_from_well = distance - well_->radius();
    double sin = std::sin(k_ * p[0]);
    double r_x_normed = (p[0] - well_->center()[0]) / distance;
    double b_nabla_a = (2/distance * distance_from_well + 2.0) * sin,
           grad_a_grad_b = 2 * distance_from_well * k_ * std::cos(k_*p[0]) * r_x_normed,
           a_nabla_b = - distance_from_well * distance_from_well * k_ * k_ * sin;
    
    return - amplitude_ * (b_nabla_a + 2*grad_a_grad_b + a_nabla_b);
}