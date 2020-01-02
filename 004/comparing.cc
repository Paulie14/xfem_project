
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

ExactBase::ExactBase()
: Function<2>(),
m_(0)
{}

ExactWellBase::ExactWellBase(Well* well, double radius, double p_dirichlet)
  : ExactBase(),
    well_(well),
    radius_(radius),
    p_dirichlet_(p_dirichlet)
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
    : ExactWellBase(well, radius, 0), k_(k), amplitude_(amplitude)
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



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////// EXACT SOLUTION 3

ExactSolution3::ExactSolution3(Well* well, double radius, double k, double amplitude)
    : ExactWellBase(well, radius, 0), k_(k), amplitude_(amplitude)
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




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////// EXACT SOLUTION 4

ExactSolution4::ExactSolution4(Well* well, double radius, double k, double amplitude)
    : ExactWellBase(well, radius, 0), k_(k), amplitude_(amplitude)
{
    double p = well_->radius()*well_->perm2aquifer(m_)*std::log(well_->radius()/radius_);
    double cs = well_->perm2aquitard(m_) / well_->perm2aquifer(m_) / well_->circumference();
    double exact_well_pressure = (cs*well_->pressure() + p_dirichlet_/(1-p))
                                /(cs + 1/(1-p)); 
    std::cout << "Exact pressure inside the well = " << exact_well_pressure << std::endl;
    
    if(well->is_active())
    {
        a_ = ( well_->perm2aquitard(m_) * well_->perm2aquifer(m_) * well_->radius() * (p_dirichlet_ - well_->pressure())) 
            / (well_->perm2aquitard(m_) + well_->perm2aquifer(m_)*well_->circumference() - p * well_->perm2aquitard(m_));
        b_ = p_dirichlet_ - a_ * std::log(radius_);
        std::cout << "Pressure on the well edge = " << this->value(well->center()) << std::endl;
    }
    else
    {
        a_ = 0;
        b_ = 0;
    }
}

Tensor< 1, 2 > ExactSolution4::grad(const Point< 2 >& p, const unsigned int component) const
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


double ExactSolution4::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
  double distance = well_->center().distance(p);
  double distance_from_well = distance - well_->radius();
  if(distance > well_->radius())
    return a_ * std::log(distance) + b_ + distance_from_well*distance_from_well * amplitude_*std::sin(k_*p[0]);
  else
    return a_ * std::log(well_->radius()) + b_;
}

double Source4::value(const Point< 2 >& p, const unsigned int /*component*/) const
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
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////// EXACT SOLUTION 5

ExactSolution5::ExactSolution5(Well* well, double amplitude)
    : ExactWellBase(well, 0, 0), amplitude_(amplitude)
{
}

void ExactSolution5::set_well_parameter(double a)
{
    a_ = a;
}


double ExactSolution5::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
  double distance = well_->center().distance(p);
  double res = 0.0;
  if(well_->is_active())
  {
    if(distance > well_->radius())
        res += a_ * std::log(distance);
    else
        res += a_ * std::log(well_->radius());
  }
  res += amplitude_ * distance * distance;
  return res;
}

Tensor< 1, 2 > ExactSolution5::grad(const Point< 2 >& p, const unsigned int component) const
{
    double distance = well_->center().distance(p);
    double distance_sqr = distance*distance;
    Tensor<1,2> grad, grad_reg;
    grad_reg[0]= (p[0] - well_->center()[0]);
    grad_reg[1]= (p[1] - well_->center()[1]);
    
    if(well_->is_active() && distance > well_->radius())
    {
        grad = a_/distance_sqr * grad_reg;
    }
    
    grad_reg = amplitude_ * 2 * grad_reg;
    return grad + grad_reg;
        
}

double Source5::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
    return - amplitude_ * 2 * 2;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////// EXACT SOLUTION 6

ExactSolution6::ExactSolution6(Well* well, double k, double amplitude)
    : ExactWellBase(well, 0, 0), k_(k), amplitude_(amplitude)
{
}

void ExactSolution6::set_well_parameter(double a)
{
    a_ = a;
}


double ExactSolution6::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
  double distance = well_->center().distance(p);
  double res = 0.0;
  if(well_->is_active())
  {
    if(distance > well_->radius())
        res += a_ * std::log(distance);
    else
        res += a_ * std::log(well_->radius());
  }
  res += amplitude_ * std::sin(distance - well_->radius());
  return res;
}

Tensor< 1, 2 > ExactSolution6::grad(const Point< 2 >& p, const unsigned int component) const
{
    double distance = well_->center().distance(p);
    Tensor<1,2> grad, grad_reg;
    grad_reg[0]= (p[0] - well_->center()[0]);
    grad_reg[1]= (p[1] - well_->center()[1]);
    
    if(well_->is_active() && distance > well_->radius())
    {
        double distance_sqr = distance*distance;
        grad = a_/distance_sqr * grad_reg;
    }
    
    grad_reg = amplitude_ / distance * std::cos(distance - well_->radius()) * grad_reg;
    return grad + grad_reg;
        
}

double Source6::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
    double res = 0.0;
    double distance = well_->center().distance(p);

//     if(distance > well_->radius())
    {
        double distance_from_well = distance - well_->radius();
        res += - amplitude_ * ( 1/distance * std::cos(distance_from_well) - std::sin(distance_from_well));
    }
    
    return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////// EXACT SOLUTION MULTIPLE

ExactSolutionMultiple::ExactSolutionMultiple(double k, double amplitude)
    : ExactBase(), k_(k), amplitude_(amplitude)
{
}

void ExactSolutionMultiple::set_wells(std::vector<Well*> wells, std::vector<double> va, std::vector<double> vb)
{
    vec_a = va;
    vec_b = vb;
    wells_ = wells;
}

Tensor< 1, 2 > ExactSolutionMultiple::grad(const Point< 2 >& p, const unsigned int component) const
{
    Tensor<1,2> res;
    for(unsigned int w=0; w<wells_.size(); w++)
    {   
        Well* well = wells_[w];
        if( ! well->is_active())
            continue;
        
        double distance = well->center().distance(p);
        double distance_sqr = distance*distance;
        Tensor<1,2> grad;
        
        if(distance > well->radius())
        {
            grad[0]= (p[0] - well->center()[0]);
            grad[1]= (p[1] - well->center()[1]);
            
            grad = vec_a[w]/distance_sqr * grad;
        }
        res += grad;
    }
    
    Tensor<1,2> grad_reg;
    grad_reg[0] = amplitude_ * k_*std::cos(k_*p[0]);
    res += grad_reg;
    
    return res;
}


double ExactSolutionMultiple::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
    double val = 0.0;
    for(unsigned int w=0; w<wells_.size(); w++)
    {
        Well* well = wells_[w];
        if( ! well->is_active())
            continue;
        
        double distance = well->center().distance(p);
        if(distance > well->radius())
        {
            val += vec_a[w] * std::log(distance);
        }
        else
            val += vec_a[w] * std::log(well->radius());
    }
    
    val += amplitude_*std::sin(k_*p[0]);
    return val;
}

SourceMultiple::SourceMultiple(double transmisivity, ExactSolutionMultiple &ex_sol)
: ExactSolutionMultiple(ex_sol.k_, ex_sol.amplitude_), transmisivity_(transmisivity)
{
    set_wells(ex_sol.wells_, ex_sol.vec_a, ex_sol.vec_b);
}

double SourceMultiple::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
    return transmisivity_ * amplitude_ * k_*k_ * std::sin(k_*p[0]);
//     double val = 0.0;
//     for(unsigned int w=0; w<wells_.size(); w++)
//     {
//         Well* well = wells_[w];
//         double distance = well->center().distance(p);
// //         double distance_from_well = distance - well_->radius();
//         if(distance > well->radius())
//             val += vec_a[w] * std::log(distance);
//         else
//             val += vec_a[w] * std::log(well->radius());
//     }
//     
//     return val + amplitude_*k_*k_*std::sin(k_*p[0]);
//     return amplitude_*k_*k_*std::sin(k_*p[0]);
    
//     double res = 0.0;
//     Well* well = wells_[0];
//     double distance = well->center().distance(p);
//     double distance_from_well = distance - well->radius();
//     if(distance > well->radius())
//         res += - amplitude_ * ( 1/distance * std::cos(distance_from_well) - std::sin(distance_from_well));
//     
//     return res;
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////// EXACT SOLUTION 7

ExactSolution7::ExactSolution7(Well* well, double k, double amplitude)
    : ExactWellBase(well, 0, 0), k_(k), amplitude_(amplitude)
{
}

void ExactSolution7::set_well_parameter(double a)
{
    a_ = a;
}


double ExactSolution7::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
  double distance = well_->center().distance(p);
  double res = 0.0;
  if(well_->is_active())
  {
    if(distance > well_->radius())
        res += a_ * std::log(distance);
    else
        res += a_ * std::log(well_->radius());
  }
  res += amplitude_*std::sin(k_*p[0]);
  return res;
}

Tensor< 1, 2 > ExactSolution7::grad(const Point< 2 >& p, const unsigned int component) const
{
    double distance = well_->center().distance(p);
    double distance_sqr = distance*distance;
    Tensor<1,2> grad, grad_reg;
    grad_reg[0]= (p[0] - well_->center()[0]);
    grad_reg[1]= (p[1] - well_->center()[1]);
    
    if(well_->is_active() && distance > well_->radius())
    {
        grad = a_/distance_sqr * grad_reg;
    }
    
    grad_reg[0] = amplitude_ * k_*std::cos(k_*p[0]);
    grad_reg[1] = 0;
    
    return grad + grad_reg;
        
}

double Source7::value(const Point< 2 >& p, const unsigned int /*component*/) const
{
    return amplitude_ * k_*k_ * std::sin(k_*p[0]);
}
