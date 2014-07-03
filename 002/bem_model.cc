
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/base/quadrature_selector.h>

//input/output of grid
#include <deal.II/grid/grid_in.h> 
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>

#include <deal.II/numerics/vector_tools.h>

//output
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>


#define _USE_MATH_DEFINES       //we are using M_PI
#include <cmath>

#include "system.hh"
#include "bem_model.hh"
#include "parameters.hh"
#include "well.hh"


BemModel::BemModel ()
  : ModelBase::ModelBase(),
    //dealii fem
    fe (1),
    dof_handler (triangulation),
    mapping(1, true),
    quadrature_formula(2),
    singular_quadrature_order(12)       //max 12 quadrature points
{
  name_ = "Default_BEM_Model";
}

BemModel::BemModel (const std::string &name,
                    const unsigned int &n_aquifers):
    ModelBase::ModelBase(name, n_aquifers),
    number_of_wells(0),
    
    //dealii fem
    fe (1),
    dof_handler (triangulation),
    mapping(1, true),
    quadrature_formula(2),
    singular_quadrature_order(12)       //max 12 quadrature points
{
}


BemModel::BemModel (const std::vector<Well*> &wells, 
                    const std::string &name,
                    const unsigned int &n_aquifers):
    ModelBase::ModelBase(wells, name, n_aquifers),
    number_of_wells(wells.size()),
    
    //dealii fem
    fe (1),
    dof_handler (triangulation),
    mapping(1, true),
    quadrature_formula(2),
    singular_quadrature_order(12)       //max 12 quadrature points
{
}

/*
BemModel::BemModel() :
    //constant
    transmisivity(1.0),
    //deal fem
    fe (1),
    dof_handler (triangulation),
    mapping(1, true),
    quadrature_formula(2),
    singular_quadrature_order(12)       //max 12 quadrature points
{  
  //setting wells
  double r = Parameters::radius;        //radius of wells
  double perm2fer = Parameters::perm2fer;
  double perm2tard = Parameters::perm2tard;
  double d = Parameters::x_dec;
  wells.push_back(Well(r,Point<2>(-d,0.0), perm2fer, perm2tard));
  wells.push_back(Well(r,Point<2>(d,0.0), perm2fer, perm2tard));
  
  //setting BC at the top of the wells
  wells[0].set_pressure(Parameters::pressure_at_top);
  wells[1].set_pressure((-1)*Parameters::pressure_at_top);
  
  number_of_wells = wells.size();
}

BemModel::BemModel(const double &permeability2fer) :
    //constant
    transmisivity(1.0),
    //deal fem
    fe (1),
    dof_handler (triangulation),
    mapping(1, true),
    quadrature_formula(2),
    singular_quadrature_order(12)       //max 12 quadrature points
{  
  //setting wells
  double r = Parameters::radius;        //radius of wells
  double perm2fer = permeability2fer;
  double perm2tard = Parameters::perm2tard;
  double d = Parameters::x_dec;
  wells.push_back(Well(r,Point<2>(-d,0.0), perm2fer, perm2tard));
  wells.push_back(Well(r,Point<2>(d,0.0), perm2fer, perm2tard));
  
  //setting BC at the top of the wells
  wells[0].set_pressure(Parameters::pressure_at_top);
  wells[1].set_pressure((-1)*Parameters::pressure_at_top);
  
  number_of_wells = wells.size();
}
*/

  
double BemModel::omega_function(const dealii::Point< 2 >& R)
{
  return (-std::log(R.norm()) / (2*numbers::PI) );
}
 
Point< 2 > BemModel::omega_normal(const dealii::Point< 2 >& R)
{
 return R / ( -2*numbers::PI * R.square()); 
}


const dealii::Quadrature< 1 >& BemModel::get_singular_quadrature(
    const DoFHandler< 1 , 2  >::active_cell_iterator& cell, const unsigned int index) const
{
  Assert(index < fe.dofs_per_cell,
            ExcIndexRange(0, fe.dofs_per_cell, index));
 
  static Quadrature<1> * q_pointer = NULL;
  if (q_pointer) delete q_pointer;
 
  q_pointer = new QGaussLogR<1>(singular_quadrature_order,
                                   fe.get_unit_support_points()[index],
                                   1./cell->measure(), true);
  return (*q_pointer);
}



void BemModel::make_grid()
{
  //trying to create 1d mesh merged from square and two circles(wells)
  //usuccesfull
  /*
  Triangulation<1,2> tria_rect;
  Triangulation<2> tria_circ1;
  Triangulation<2> tria_circ2;
    
  GridGenerator::hyper_rectangle<1,2>(tria_rect,down_left,up_right);
  
  
  GridGenerator::hyper_ball<2>(tria_circ1,wells[0].GetPosition(),wells[0].GetRadius());
  GridGenerator::hyper_ball<2>(tria_circ2,wells[1].GetPosition(),wells[1].GetRadius());
  
  static const HyperBallBoundary<1,2> boundary_circ1(wells[0].GetPosition(),wells[0].GetRadius());
  tria_circ1.set_boundary(1, boundary_circ1);
  
  static const HyperBallBoundary<1,2> boundary_circ2(wells[1].GetPosition(),wells[1].GetRadius());
  tria_circ2.set_boundary(1, boundary_circ2);
  
  
  GridGenerator::merge_triangulations<1,2>(tria_rect,tria_circ1,tria_rect);
  GridGenerator::merge_triangulations<1,2>(tria_rect,tria_circ2,triangulation);
  
  StraightBoundary<1,2> boundary_r();
  Triangulation<1,2> tria_rect_1;
  Triangulation<1,2> tria_rect_2;
  Triangulation<1,2> tria_rect_3;
  Triangulation<1,2> tria_rect_4;
  Triangulation<1,2> tria_rect_temp_1;
  Triangulation<1,2> tria_rect_temp_2;
  
  GridGenerator::hyper_rectangle<1,2>(tria_rect_1,down_left,down_right);
  GridGenerator::hyper_rectangle<1,2>(tria_rect_2,down_right,up_right);
  
  GridGenerator::hyper_rectangle<1,2>(tria_rect_3,up_left,up_right);
  GridGenerator::hyper_rectangle<1,2>(tria_rect_4,up_left,down_right);
  
  std::cout << down_left << "\n" <<
               down_right << "\n" <<
               up_left << "\n" <<
               up_right << "\n" << std::endl;
  
  GridGenerator::merge_triangulations<1,2>(tria_rect_1,tria_rect_2,tria_rect_temp_1);
  //GridGenerator::merge_triangulations<1,2>(tria_rect_3,tria_rect_4,tria_rect_temp_2);
  //GridGenerator::merge_triangulations<1,2>(tria_rect_4,tria_rect_temp_1,triangulation);
  GridGenerator::merge_triangulations<1,2>(tria_rect_temp_1,tria_rect_3,triangulation);
  
  static const HyperBallBoundary<1,2> boundary_circ1(wells[0].GetPosition(),wells[0].GetRadius());
  static const HyperBallBoundary<1,2> boundary_circ2(wells[1].GetPosition(),wells[1].GetRadius());
  
  triangulation.set_boundary(1,boundary_circ1);
  triangulation.set_boundary(2,boundary_circ2);
  //*/
  
  //open filestream with mesh from GMSH
  std::ifstream in;
  in.open(bem_mesh_file);
  
  GridIn<1,2> gridin;
  //attaching object of triangulation
  gridin.attach_triangulation(triangulation);
  //reading data from filestream
  gridin.read_msh(in);
}

void BemModel::setup_system()
{
  dof_handler.distribute_dofs(fe);
 
  const unsigned int n_dofs =  dof_handler.n_dofs();
  
  matrix_size = n_dofs+number_of_wells;
  //we have one more equation for the mean value of the pressure head in the wells
  system_matrix.reinit(matrix_size, matrix_size);
  system_rhs.reinit(matrix_size);
  bem_solution.reinit(matrix_size); 
  alpha.reinit(n_dofs); 
  
  std::cout << "number of degrees of freedom: " << n_dofs << std::endl;
  std::cout << "number of equaitons (matrix size): " << matrix_size << std::endl;
}

void BemModel::assemble_system()
{
  FEValues<1,2> fe_v(mapping, fe, quadrature_formula,
                           update_values |
                           update_cell_normal_vectors |
                           update_quadrature_points |
                           update_JxW_values);
  
 
  const unsigned int n_q_points = fe_v.n_quadrature_points;

  std::vector<unsigned int> local_dof_indices(fe.dofs_per_cell);

  Vector<double>      local_matrix_row_i(fe.dofs_per_cell);


  std::vector<Point<2> > support_points(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points<1,2>( mapping, dof_handler, support_points);

  /*
  for (int i = 0; i < dof_handler.n_dofs(); i++)
    std::cout << i << ":  " << support_points[i] << std::endl;
  //*/
  
  //addition to equation for mean value of pressure head on well
  number_of_well_elements = 0;

  
  typename DoFHandler<1,2>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  
  //iteration over cells - in row
  for (; cell != endc; ++cell)
    {
      fe_v.reinit(cell);
      cell->get_dof_indices(local_dof_indices);   
      
      //std::cout << cell->index() << ":   mat.ID:  " << cell->material_id() << std::endl; 
    
      const std::vector<Point<2> > &q_points = fe_v.get_quadrature_points();
      const std::vector<Point<2> > &normals = fe_v.get_normal_vectors();
      
      unsigned int c_mat = cell->material_id();
      unsigned int c_i = cell->index();
      
      //counting elements on well
      if(c_mat == 2 ) number_of_well_elements++;
      
      //alpha is an internal angle divided by 2PI
      if(is_corner_point(support_points[c_i])) 
        {
          alpha[c_i] = 0.125;
        }
      else
        alpha[c_i] = 0.5;
        
      
      system_matrix(c_i, c_i) += alpha(c_i);
      
      //std::cout << alpha[c_i] << std::endl;
      
      switch (c_mat)
        {
          case 1: //square domain
          {
            //filling RHS and well equations for the square
            //Neumann BC on square
            system_rhs(c_i) = 0.0;
            //zeros on square in equations for mean value of the pressure head in the wells
            for(unsigned int k=0; k < number_of_wells; k++)
              system_matrix(c_i,dof_handler.n_dofs()+k) = 0.0;
          } break;
          
          case 2: //first well
          case 3: //second well
          {
            //filling equations for mean value of the pressure head in the wells
            for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
              system_matrix(dof_handler.n_dofs() + c_mat-2,local_dof_indices[j]) = 1;
          } break;
        }
      
      //iteration in columns
      for (unsigned int i=0; i<dof_handler.n_dofs() ; ++i)
        {
          local_matrix_row_i = 0;

          bool is_singular = false;
          unsigned int singular_index = numbers::invalid_unsigned_int;

          //finding singular point on current cell
          for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
            if (local_dof_indices[j] == i)
              {
                singular_index = j;
                is_singular = true;
                break;
              }

          if (is_singular == false)
            {
              for (unsigned int q=0; q<n_q_points; ++q)
                {
                  //getting vector R 
                  const Point<2> R = q_points[q] - support_points[i];
                  
                  for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
                  {
                    //doing q = sigma*(H - u_average)
                    switch (c_mat)
                    {                    
                      case 2:
                      case 3:
                      {
                        //number of proper well according to material id
                        unsigned int w = c_mat - 2;
                        //RHS except the mean value of pressure on the well bc = sigma*H
                        system_rhs(c_i) -= ( wells[w]->perm2aquifer()  *
                                             wells[w]->pressure()  *
                                             fe_v.shape_value(j,q)         *
                                             omega_function(R)              *
                                             fe_v.JxW(q) );
                      
                        //addition of the mean value of the pressure on the well = sigma*u_average
                        system_matrix(c_i,dof_handler.n_dofs() + w)
                                        -= ( wells[w]->perm2aquifer() * 
                                             omega_function(R)           *
                                             fe_v.shape_value(j,q)        *
                                             fe_v.JxW(q) );  
                      }     
                    }//switch
                     
                    //add 2 integrals into 2 main matrix elements = a_ij^alpha (BEM)
                    local_matrix_row_i(j) += ( ( omega_normal(R)     *
                                                  normals[q] )         *
                                                  fe_v.shape_value(j,q) *
                                                  fe_v.JxW(q)       );
                  } //for(j)
                } //for(q)
            } //if(singular) 
            else {
            Assert(singular_index != numbers::invalid_unsigned_int,
                     ExcInternalError());
          
            const Quadrature<1> & singular_quadrature =
              get_singular_quadrature(cell, singular_index);

            FEValues<1,2> fe_v_singular (mapping, fe, singular_quadrature,
                                                 update_jacobians |
                                                 update_values |
                                                 update_cell_normal_vectors |
                                                 update_quadrature_points );

            fe_v_singular.reinit(cell);

            const std::vector<Point<2> > &singular_normals = fe_v_singular.get_normal_vectors();
            const std::vector<Point<2> > &singular_q_points = fe_v_singular.get_quadrature_points();

            for (unsigned int q=0; q<singular_quadrature.size(); ++q)
              {
                const Point<2> R = singular_q_points[q] - support_points[i];
                
                for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
                  {
                    //doing q = sigma*(H - u_average)
                    switch (c_mat)
                    {                    
                      case 2:
                      case 3:
                      {
                        //number of proper well according to material id
                        unsigned int w = c_mat - 2;
                        //RHS except the mean value of pressure on the well bc = sigma*H
                        system_rhs(c_i) -= ( wells[w]->perm2aquifer()   *
                                             wells[w]->pressure()   *
                                             fe_v_singular.shape_value(j,q) *
                                             omega_function(R)               *
                                             fe_v_singular.JxW(q) );
                      
                        //addition of the mean value of the pressure on the well = sigma*u_average
                        system_matrix(c_i,dof_handler.n_dofs() + w)
                                        -= ( wells[w]->perm2aquifer()   * 
                                             omega_function(R)             *
                                             fe_v_singular.shape_value(j,q) *
                                             fe_v_singular.JxW(q) );  
                      }     
                    }//switch
                     
                    //add 2 integrals into 2 main matrix elements = a_ij^alpha (BEM)
                    local_matrix_row_i(j) += ( ( omega_normal(R)               *
                                                  singular_normals[q] )         *
                                                  fe_v_singular.shape_value(j,q) *
                                                  fe_v_singular.JxW(q) );
                  } //for(j)
                } //end of for(q)
              } //end of else
            
            for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
              system_matrix(i,local_dof_indices[j])
                += local_matrix_row_i(j);

      } //end of for over columns
    } //end of cell iteration
 
    //addition to equation for mean value of pressure head on well
    for (unsigned int i = 0; i < number_of_wells; i++)
      system_matrix(dof_handler.n_dofs()+i,dof_handler.n_dofs()+i) = number_of_well_elements*(-1.0); 
    
    //printing system matrix
    //system_matrix.print(std::cout);
}


void BemModel::solve()
{
  SolverControl solver_control(system_matrix.m(),1e-12*system_rhs.l2_norm());
  SolverGMRES<Vector<double> > solver (solver_control);
  solver.solve (system_matrix, bem_solution, system_rhs, PreconditionIdentity());
  
  std::cout << std::scientific << "Solver: steps: " << solver_control.last_step() << "\t residuum: " << setprecision(4) << solver_control.last_value() << std::endl;
  
  std::cout << "mean pressure1:\t" << bem_solution[bem_solution.size()-2] << std::endl;
  std::cout << "mean pressure2:\t" << bem_solution[bem_solution.size()-1] << std::endl;
}


dealii::Vector< double > BemModel::get_boundary_solution(const std::vector< Point< 2 > >& points)
{
  //point - difference between points
  Point<2> p;
  //distance from the closest support point
  double eps = Parameters::radius*M_PI / number_of_well_elements;
  
  unsigned int n_points = points.size();
  
  //returning solution
  dealii::Vector<double> solution(n_points);
  
  std::vector<Point<2> > support_points(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points<1,2>( mapping, dof_handler, support_points);
  
  
  for (unsigned int i = 0; i < support_points.size(); i++)
  {
    for(unsigned int j = 0; j < points.size(); j++)
    {
      p = support_points[i] - points[j]; 
      if(p.norm() < eps)
        {
          solution[j] = bem_solution[i];
          //  *fe_v.shape_value(j,q)*fe_v.JxW(q);
          //std::cout << support_points[i] << "\t" << points[j] << "\t" << p.norm()
          //          << std::setw(5) << i << "\t" << solution[j] << std::endl;
        }
    }
  }    
  
  return solution;
}


const dealii::Vector< double >& BemModel::get_distributed_solution()
{
  MASSERT(0,"Is not implemented in class BemModel.");
  return bem_solution; 
}


const dealii::Vector< double >& BemModel::get_solution()
{
  MASSERT(bem_solution.size() < 1,"Solution has not been computed yet.");
  return bem_solution; 
}



void BemModel::get_solution_at_points(const std::vector< Point< 2 > > &points, Vector<double> &solution)
{
  
  DBGMSG("get_solution_at_points\n");
  
  unsigned int n_points = points.size();
  
  solution.reinit(n_points); //initialize and puts zeros everywhere
  
  FEValues<1,2> fe_v(mapping, fe, quadrature_formula,
                           update_values |
                           update_cell_normal_vectors |
                           update_quadrature_points |
                           update_JxW_values);
  
  const unsigned int n_q_points = fe_v.n_quadrature_points;
   
  std::vector<Point<2> > support_points(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points<1,2>( mapping, dof_handler, support_points);

  typename DoFHandler<1,2>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
      
  //q (flow) boundary condition according to the type of boundary in the integral
  std::vector<double> q_ja(number_of_wells+1);
  q_ja[0] = 0.0;
  for(unsigned int w=0; w < number_of_wells; w++)
  {
    q_ja[w+1] = wells[w]->perm2aquifer() * ( wells[w]->pressure() -
                                               bem_solution[dof_handler.n_dofs() + w] );
  }
  
  //iteration over cells - on each cell computing addition to every point
  for (; cell != endc; ++cell)
  {
    std::cout << "\b\b\b\b\b\b" << std::setw(6) << cell->index();
    fe_v.reinit(cell);
    const std::vector<Point<2> > &q_points = fe_v.get_quadrature_points();
    const std::vector<Point<2> > &normals = fe_v.get_normal_vectors();
      
    unsigned int c_mat = cell->material_id();
    unsigned int c_i = cell->index();
    

    //iterating over points
    for (unsigned int i = 0; i < n_points; i++)
    {  
      //iteration over quadrature points (integrating)
      for (unsigned int q=0; q<n_q_points; ++q)
        {
          //getting vector R 
          const Point<2> R = q_points[q] - points[i];               
   
            //std::cout << c_mat << "\t" << R.norm() << std::endl;
            for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
            {
              //std::cout << "--"; 
              solution[i] += ( q_ja[c_mat-1]       * 
                               omega_function(R)    * 
                               fe_v.shape_value(j,q) *
                               fe_v.JxW(q));
              
              solution[i] -= ( bem_solution[c_i+j] * //u_j,alpha       
                                  ( omega_normal(R)  *            //omega normal derivate
                                    normals[q] )       *          //in distance(vector) R
                                    fe_v.shape_value(j,q)*        //phi_alpha
                                    fe_v.JxW(q) );
            }
        }
    }
  }
  std::cout << std::endl;
}


void BemModel::output_distributed_solution(const std::string &mesh_file, const std::string &flag_file, const unsigned int &cycle, const unsigned int &m_aquifer)
{
  //triangulation for distributing solution onto domain
  Triangulation<2> dist_tria;
  QGauss<2>        dist_quadrature(2);
  FE_Q<2>          dist_fe(1);                    
  DoFHandler<2>    dist_dof_handler(dist_tria);
  FEValues<2>      dist_fe_values(dist_fe, dist_quadrature,
                                  update_values | update_gradients | update_JxW_values);

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
  
  //====================computing solution on the triangulation
  std::vector< Point< 2 > > support_points;
  Vector<double> dist_solution;
  support_points.resize(dist_dof_handler.n_dofs());
  dist_solution.reinit(dist_dof_handler.n_dofs());
  
  DoFTools::map_dofs_to_support_points<2>(dist_fe_values.get_mapping(), dist_dof_handler, support_points);
  std::cout << "...computing solution on:   " << mesh_file << std::endl;
  std::cout << "...number of dofs in the mesh:   " << dist_dof_handler.n_dofs() << std::endl;
  std::cout << "...number of dofs in the BEM mesh:   " << dof_handler.n_dofs() << std::endl;
  
  get_solution_at_points(support_points, dist_solution);
  
  //====================vtk output
  DataOut<2> data_out;
  data_out.attach_dof_handler (dist_dof_handler);
  data_out.add_data_vector (dist_solution, "solution");
  data_out.build_patches ();

  std::stringstream filename;
  filename << output_dir_ << "bem_distributed_" << cycle << ".vtk";
   
  std::ofstream output (filename.str());
  data_out.write_vtk (output);
  
  std::cout << "\noutput written in:\t" << filename.str() << std::endl; 
}


void BemModel::run(const unsigned int cycle)
{
  std::cout << "===== BEM Model start   " << cycle << "=====" << std::endl;
  make_grid();
  setup_system();
  assemble_system();
  solve();
  std::cout << "===== BEM Model finihed   " << cycle << "=====" << std::endl;
}


void BemModel::output_results(const unsigned int cycle)
{
  MASSERT(0, "Is not implemented - use output_2d_results() and set mesh before!");
}




//#########################################################################
//========================= helpful functions =============================
//#########################################################################


bool BemModel::is_corner_point(Point< 2 > point)
{
  //corner points of the square for defining alpha
  std::vector<Point<2> > corner_points(4);
  corner_points[0] = Point<2>((-1.0)*Parameters::sqr,(-1.0)*Parameters::sqr);
  corner_points[1] = Point<2>((-1.0)*Parameters::sqr,Parameters::sqr);
  corner_points[2] = Point<2>(Parameters::sqr,(-1.0)*Parameters::sqr);
  corner_points[3] = Point<2>(Parameters::sqr,Parameters::sqr);
  
  for(unsigned int w = 0; w < corner_points.size(); w++)
    {
      if(corner_points[w] == point) return true;
    }
  
  return false;
}


void BemModel::write_bem_solution()
{
  typename DoFHandler<1,2>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
      
  for (; cell!=endc; ++cell)
  {
    std::cout << std::setw(5) << cell->index() << "\t" 
              << cell->material_id() << "\t"
              << bem_solution[cell->index()] <<std::endl;
  }  
}

void BemModel::get_boundary_elm_index(const std::vector<dealii::Point<2 > > & points, 
                                       std::vector<unsigned int> & indexes)
{
  //point - difference between points
  Point<2> p;
  //distance from the closest support point
  double eps = Parameters::radius*M_PI / number_of_well_elements;
  
  //writing in vector of indexes
  indexes.resize(points.size());
  
  std::vector<Point<2> > support_points(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points<1,2>( mapping, dof_handler, support_points);
  
  
  for (unsigned int i = 0; i < support_points.size(); i++)
  {
    for(unsigned int j = 0; j < points.size(); j++)
    {
      p = support_points[i] - points[j]; 
      if(p.norm() < eps)
        {
          std::cout << points[j](0) << "\t" << points[j](1) << "\t" << p.norm() <<std::endl;
          indexes[j] = i;
        }
    }
  }
}


















