
#include <deal.II/grid/grid_in.h> 
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/persistent_tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/base/table_handler.h>
#include <deal.II/base/convergence_table.h>

#include "system.hh"
#include "model.hh"
#include "xmodel.hh"
#include "simple_models.hh"
#include "bem_model.hh"
#include "comparing.hh"
#include "well.hh"
#include "parameters.hh"
#include "exact_model.hh"

#include "adaptive_integration.hh"

#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <iostream>

using namespace std;


class TestIntegration : public Function<2>
    {
      public:
        ///Constructor
        TestIntegration(Well* well) : Function< 2 >()
        {
            this->well = well;
        }
        
        ///Returns the value of pressure at the boundary.
        virtual double value (const Point<2>   &p,
                              const unsigned int  component = 0) const
        {
//             double distance = std::abs(p[0]-well->center()[0]);
//             if (distance <= well->radius())
//                 return std::log(well->radius());
//   
//             return std::log(distance);
            double distance = p.distance(well->center());
            if (distance <= well->radius())
                return std::log(well->radius());
            
            distance = std::log(distance);
            if(distance >=0) return 0;
            
            return distance;
        }
    private:
        Well *well;
    };

class TestIntegration_r2 : public Function<2>
    {
      public:
        ///Constructor
        TestIntegration_r2(Well* well) : Function< 2 >()
        {
            this->well = well;
        }
        
        ///Returns the value of pressure at the boundary.
        virtual double value (const Point<2>   &p,
                              const unsigned int  component = 0) const
        {
            double distance = p.distance(well->center());
            if (distance <= well->radius())
                return 0.0;
            else
                return 1/(distance*distance);
        }
    private:
        Well *well;
    };
    
class WellCharacteristicFunction : public Function<2>
    {
      public:
        ///Constructor
        WellCharacteristicFunction(Well* well) : Function< 2 >()
        {
            this->well = well;
        }
        
        ///Returns the value of pressure at the boundary.
        virtual double value (const Point<2>   &p,
                              const unsigned int  component = 0) const
        {
            double distance = p.distance(well->center());
            if (distance <= well->radius())
                return 0.0;
            else return 1.0;
        }
    private:
        Well *well;
    };
    
class Dirichlet_pressure : public Function<2>
    {
      public:
        ///Constructor
        Dirichlet_pressure(const Point<2> &down_left, const Point<2> &up_right) 
          : Function< 2 >(),
            down_left(down_left),
            up_right(up_right)
        {}
        
        ///Returns the value of pressure at the boundary.
        virtual double value (const Point<2>   &p,
                              const unsigned int  component = 0) const
        {
          if( (p[0] == down_left[0]) || (p[0] == up_right[0]) )
            return p[1] - up_right[1];
          
          //zero at top
          if(p[1] == up_right[1])
            return 0.0; 
          
          //in unspecified case
          return 0.0;
        }
      
      private:  
        Point<2> down_left, //coordinates of the down_left corner
                 up_right;  //coordinates of the up_right corner
    };
    
class Dirichlet_piezo_const : public Function<2>
    {
      public:
        ///Constructor
        Dirichlet_piezo_const() : Function< 2 >()
        {}
        
        ///Returns the value of pressure at the boundary.
        virtual double value (const Point<2>   &p,
                              const unsigned int  component = 0) const
        {
          return 0.0;
        }
      
      private:  
    };
    


void test_squares()
{
    Point<2> a(2, 3),
             b(5, 7);
    
    Square square(a,b);
    
    for(Point<2> &p:  square.vertices)
    {
        std::cout << p[0] << "   " << p[1] << std::endl;
    }
    square.mapping.print(std::cout);
    std::cout << "jacobian = " << square.mapping.jakobian() << std::endl;
    std::cout << "inverse jacobian = " << square.mapping.jakobian_inv() << std::endl;
    
    Point<2> test_point1(4,5),
             test_point2(0,0);
    std::cout << "test_point: " << test_point1 << std::endl;
    test_point2 = square.mapping.map_real_to_unit(test_point1);
    std::cout << test_point2 << std::endl;
    test_point2 = square.mapping.map_unit_to_real(test_point2);
    std::cout << test_point2 << std::endl;
}

    
void bedrichov_tunnel()
{
  //area of the model
  //coordinates of the down_left corner
  Point<2> down_left(0,-300);
  //coordinates of the up_right corner
  Point<2> up_right(500,0);
  //coordinates of the tunnel center
  Point<2> tunnel_center(250,-120);
  
  double tunnel_radius = 1.8,
         tunnel_sigma = 1e5,
         bulk_transmisivity = 1e-8;
  
  std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
  
  Well *tunnel = new Well( tunnel_radius,
                           tunnel_center,
                           tunnel_sigma, 
                           1e10);
  tunnel->set_pressure(tunnel_center[1]);
  tunnel->evaluate_q_points(500);
  
  Dirichlet_pressure *dirichlet = new Dirichlet_pressure(down_left, up_right);
  Dirichlet_piezo_const *dirichlet_piezo_const = new Dirichlet_piezo_const();
  
  //FEM model creation
  Model_simple tunnel_adapt_linear(tunnel);  
  tunnel_adapt_linear.set_area(down_left,up_right);
  tunnel_adapt_linear.set_name("tunnel_adapt_linear");
  tunnel_adapt_linear.set_output_dir("../output");
  tunnel_adapt_linear.set_transmisivity(bulk_transmisivity,0);
  tunnel_adapt_linear.set_initial_refinement(1);  
  tunnel_adapt_linear.set_ref_coarse_percentage(0.3,0.05);
  tunnel_adapt_linear.set_adaptivity(true);
  //tunnel_adapt_linear.set_dirichlet_function(dirichlet);
  tunnel_adapt_linear.set_dirichlet_function(dirichlet_piezo_const);
  
  for (unsigned int cycle=0; cycle < 13; ++cycle)
    { 
      std::cout << "===== Model running   " << cycle << "   =====" << std::endl;
      tunnel_adapt_linear.run (cycle);
      tunnel_adapt_linear.output_results (cycle);
      std::cout << "===== Model finished =====" << std::endl;
    }
  
  
  XModel_simple tunnel_xfem(tunnel);  
  tunnel_xfem.set_area(down_left,up_right);
  tunnel_xfem.set_name("tunnel_xfem");
  tunnel_xfem.set_output_dir("../output");
  tunnel_xfem.set_transmisivity(bulk_transmisivity,0);
  tunnel_xfem.set_initial_refinement(4);                                     
  tunnel_xfem.set_enrichment_radius(50);
  tunnel_xfem.set_dirichlet_function(dirichlet_piezo_const);
  
  tunnel_xfem.run();     
  tunnel_xfem.output_distributed_solution(tunnel_adapt_linear.get_triangulation());
    
  delete dirichlet;
  delete tunnel;
}
  
void test_circle_grid_creation(const std::string &input_dir)
{
  /** *******************************
    * Trying circle grid
    * *******************************/
  //PARAMETERS
  //Point<2> down_left((-2)*Parameters::sqr,(-2)*Parameters::sqr);
  //Point<2> up_right(2*Parameters::sqr,2*Parameters::sqr);
  unsigned int global_ref = 1,
               boundary_ref = 3;
  
  //creation of coarse
  Triangulation<2> coarse_tria;
  //Point<2> center((down_left+up_right)/2);
  //double radius =  down_left.distance(up_right) / 2;
  Point<2> center(0,0);
  double radius =  10;
  GridGenerator::hyper_ball<2>(coarse_tria,center,radius);
  static const HyperBallBoundary<2> boundary(center,radius);
  coarse_tria.set_boundary(0, boundary);
      
  PersistentTriangulation<2> *triangulation = new PersistentTriangulation<2>(coarse_tria);
  triangulation->restore();    
  triangulation->refine_global(global_ref);
  
  Triangulation<2>::active_cell_iterator cell, endc;  
  for(unsigned int ref=0; ref < boundary_ref; ref++)
  {
    cell = triangulation->begin_active();
    endc = triangulation->end();
    //DBGMSG("refinement: %d\n",ref);
    for (; cell!=endc; ++cell)
    {
      if(cell->at_boundary())
      {
        cell->set_refine_flag();
      }
    }
    triangulation->execute_coarsening_and_refinement();
  }
  
  std::stringstream filename_real, filename_coarse, filename_flags;
  
  filename_real << input_dir + "circle_grid_1.msh";
  filename_coarse << input_dir + "circle_grid_coarse_1.msh";
  filename_flags << input_dir + "circle_grid_flags_1.ptf";
  
  std::ofstream output(filename_coarse.str().c_str());
  
  GridOut grid_out;
  
  grid_out.write_msh<2> (coarse_tria, output);
  
  output.clear();
  output.close();
  output.open(filename_real.str().c_str());
  grid_out.write_msh<2> (*triangulation, output);
   
  output.clear();
  output.close();
  output.open(filename_flags.str().c_str());
  
  triangulation->write_flags(output);
  output.close();
  
  delete triangulation; 
  //*/
}

/*
void test_convergence_square(std::string output_dir)
{
  std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE ::::::::::::::::\n\n" << std::endl;
  
  //------------------------------SETTING----------------------------------
  std::string test_name = "square_convergence_";
  bool fem_create = false,
       fem = false,
       xfem = true,
       ex = false;
  
  double p_a = 10.0,    //area of the model
         p_b = 10.0,
         excenter = 0.0, //0.05,
         radius = p_a*std::sqrt(2),
         well_radius = 0.02,
         perm2fer = Parameters::perm2fer, 
         perm2tard = Parameters::perm2tard,
         transmisivity = Parameters::transmisivity,
         enrichment_radius = 4.0;
         
  unsigned int n_well_q_points = 500;
         
  Point<2> well_center(0,0);
  
  //std::string input_dir = "../input/square_convergence/";
  std::string input_dir = "../input/square_convergence/";
  std::string coarse_file = input_dir + "coarse_grid.msh";
  //std::string coarse_file_0 = input_dir + "coarse_grid.msh";
  //std::string ref_flags_coarse = input_dir + "ref_flags_1.ptf";
  std::string ref_flags_fine = input_dir + "ref_flags_7.ptf";
  
  PersistentTriangulation<2>* fine_triangulation = NULL;
  //--------------------------END SETTING----------------------------------
  
  Point<2> down_left(-p_a+excenter,-p_a+excenter);
  Point<2> up_right(p_a+excenter, p_a+excenter);
  std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
  
  
  Well *well = new Well( well_radius,
                         well_center,
                         perm2fer, 
                         perm2tard);
  well->set_pressure(Parameters::pressure_at_top);
  well->evaluate_q_points(n_well_q_points);
  
  //the radius is the half of the diagonal of the square: 2*p_a*sqrt(2)/2 = p_a*sqrt(2)
  Function<2> *dirichlet_square = new Dirichlet_exact_square(well,radius);
  
  //FEM model creation
  Model_simple model_simple(well);  
  model_simple.set_name(test_name + "fem_model");
  model_simple.set_output_dir(output_dir);
  model_simple.set_area(down_left,up_right);
  model_simple.set_transmisivity(transmisivity,0);
  model_simple.set_initial_refinement(1);  
  model_simple.set_ref_coarse_percentage(0.95,0.05);
  //model_simple.set_ref_coarse_percentage(0.3,0.05);
  //model_simple.set_grid_create_type(ModelBase::rect);
  
  model_simple.set_grid_create_type(ModelBase::load);
  model_simple.set_computational_mesh(coarse_file);
  model_simple.set_dirichlet_function(dirichlet_square);
  model_simple.set_adaptivity(true);
  model_simple.set_matrix_output(false);
  
  XModel_simple xmodel_simple(well);  
  xmodel_simple.set_name(test_name + "sgfem_model");
  xmodel_simple.set_enrichment_method(Enrichment_method::sgfem);
  //xmodel_simple.set_name(test_name + "xfem_shift_model");
  //xmodel_simple.set_enrichment_method(Enrichment_method::xfem_shift);
  
  xmodel_simple.set_output_dir(output_dir);
  xmodel_simple.set_area(down_left,up_right);
  xmodel_simple.set_transmisivity(transmisivity,0);
  xmodel_simple.set_initial_refinement(1);                                     
  xmodel_simple.set_enrichment_radius(enrichment_radius);
  xmodel_simple.set_grid_create_type(ModelBase::rect);
  //xmodel_simple.set_computational_mesh(coarse_file, ref_flags_coarse);
  xmodel_simple.set_dirichlet_function(dirichlet_square);
  xmodel_simple.set_adaptivity(true);
  //xmodel_simple.set_enrichment_method(Enrichment_method::xfem_shift);
  //xmodel_simple.set_well_computation_type(Well_computation::sources);
  //xmodel_simple.set_output_features(true, true);
  xmodel_simple.set_matrix_output(true);
   
//   // creation of the very fine mesh
//   // uncomment if you want to create the fine mesh, e.g. for exact solution and comparision
//   if(fem_create)
//     
//   {
//     model_simple.set_grid_create_type(ModelBase::rect);
//     model_simple.set_initial_refinement(7);
//     model_simple.set_ref_coarse_percentage(0.2,0.0);
//     for (unsigned int cycle=0; cycle < 15; ++cycle)
//     { 
//       std::cout << "===== Model running   " << cycle << "   =====" << std::endl;
//       model_simple.run (cycle);
//       model_simple.output_results (cycle);
//       std::cout << "===== Model finished =====" << std::endl;
//     }
//     return;
//   }

  
  
//   { //MESH CREATION
//   // open fine mesh -------------------------------------------------
//   std::cout << "creating/restoring fine mesh..." << std::endl;
//   Triangulation<2> coarse_tria;
//   std::ifstream in;
//   GridIn<2> gridin;
//   in.open(coarse_file);
//   //attaching object of triangulation
//   gridin.attach_triangulation(coarse_tria);
//   if(in.is_open())
//   {
//     //reading data from filestream
//     gridin.read_msh(in);
//   }          
//   else
//   {
//     xprintf(Err, "Could not open coarse grid file: %s", coarse_file.c_str());
//   }
//   in.close();
//   in.clear();
//   
//   
//   fine_triangulation = new PersistentTriangulation<2>(coarse_tria);
//   fine_triangulation->restore();
//   for(unsigned int i = 0; i <6; i++)
//   {
//     fine_triangulation->set_all_refine_flags();
//     fine_triangulation->execute_coarsening_and_refinement();
//   }
//    
//    //output of refinement flags of persistent triangulation
//    std::stringstream ref_flags_start_file;
//    ref_flags_start_file << output_dir << "ref_flags_start" << ".ptf";
//    std::ofstream output (ref_flags_start_file.str());
//    //output.open(ref_flags_start_file.str());
//    fine_triangulation->write_flags(output);
//    
//   // creation of the very fine mesh
//   // uncomment if you want to create the fine mesh, e.g. for exact solution and comparision
//   
//     model_simple.set_name(test_name + "fem_model_mesh_creation");
//     model_simple.set_grid_create_type(ModelBase::load);
//     model_simple.set_computational_mesh(coarse_file, ref_flags_start_file.str());
//     model_simple.set_ref_coarse_percentage(0.2,0.0);
//     for (unsigned int cycle=0; cycle < 15; ++cycle)
//     { 
//       std::cout << "===== Model running   " << cycle << "   =====" << std::endl;
//       model_simple.run (cycle);
//       model_simple.output_results (cycle);
//       std::cout << "===== Model finished =====" << std::endl;
//     }
//     return;
//   
//   }
  
  
  // open fine mesh -------------------------------------------------
  std::cout << "creating/restoring fine mesh..." << std::endl;
  Triangulation<2> coarse_tria;
  std::ifstream in;
  GridIn<2> gridin;
  in.open(coarse_file);
  //attaching object of triangulation
  gridin.attach_triangulation(coarse_tria);
  if(in.is_open())
  {
    //reading data from filestream
    gridin.read_msh(in);
  }          
  else
  {
    xprintf(Err, "Could not open coarse grid file: %s", coarse_file.c_str());
  }
        
  fine_triangulation = new PersistentTriangulation<2>(coarse_tria);
  in.close();
  in.clear();
  in.open(ref_flags_fine);
  if(in.is_open())
    fine_triangulation->read_flags(in);
  else
  {
    xprintf(Warn, "Could not open refinement flags file: %s\n Ingore this if loading mesh without refinement flag file.\n", ref_flags_fine.c_str());
  }
  //creates actual grid to be available
  fine_triangulation->restore();
  //end reading fine mesh -------------------------------------------
  
  // Exact model
  if(ex)
  {
    std::cout << "computing exact solution on fine mesh..." << std::endl;
    ExactModel exact(well, radius);
    exact.output_distributed_solution(*fine_triangulation);
    ExactBase* exact_solution = new ExactSolution(well, radius);
    double exact_norm = Comparing::L2_norm_exact(*fine_triangulation,exact_solution);
    std::cout << "L2_norm of the exact solution: " << exact_norm << std::endl;
    //return;
  }
  
  
  unsigned int n_cycles = 15;
  double l2_norm_dif_xfem, l2_norm_dif_fem;
  
  TableHandler table_convergence;
  
  ExactBase* exact_solution = new ExactSolution(well, radius);
  std::cout << "L2_norm of the exact solution: " << 
      Comparing::L2_norm_exact(*fine_triangulation,exact_solution) << std::endl;
  
  for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
      table_convergence.add_value("Cycle",cycle);
      if(fem)
      {
      std::cout << "===== FEM Model_simple running   " << cycle << "   =====" << std::endl;
      model_simple.run (cycle);
      model_simple.output_distributed_solution(*fine_triangulation, cycle);
      //model_simple.output_results (cycle);
      l2_norm_dif_fem = Comparing::L2_norm_diff( model_simple.get_distributed_solution(),
                                                 *fine_triangulation,
                                                 exact_solution);
      
      table_convergence.add_value("$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$",l2_norm_dif_fem);
      table_convergence.set_precision("$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$", 2);
      table_convergence.set_scientific("$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$",true);
      
      table_convergence.add_value("FEM-dofs",model_simple.get_number_of_dofs());
      table_convergence.add_value("It_{FEM}",model_simple.solver_iterations());
      
      table_convergence.add_value("FEM-time",model_simple.get_last_run_time());
      table_convergence.set_precision("FEM-time", 3);
      std::cout << "===== FEM Model_simple finished =====" << std::endl;
      }
      
      
      if(xfem)
      {
      //if(cycle < 5)
      {
      std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
      xmodel_simple.run (cycle);  
      xmodel_simple.output_distributed_solution(*fine_triangulation,cycle);
      std::cout << "===== XModel_simple finished =====" << std::endl;
      
      l2_norm_dif_xfem = Comparing::L2_norm_diff( xmodel_simple.get_distributed_solution(),
                                                  *fine_triangulation,
                                                  exact_solution);
      }
      
      table_convergence.add_value("$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$",l2_norm_dif_xfem);
      table_convergence.set_precision("$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$", 2);
      table_convergence.set_scientific("$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$",true);
      
      table_convergence.add_value("XFEM-dofs",xmodel_simple.get_number_of_dofs().first+xmodel_simple.get_number_of_dofs().second);
      table_convergence.add_value("XFEM-enriched dofs",xmodel_simple.get_number_of_dofs().second);
      table_convergence.add_value("It_{XFEM}",xmodel_simple.solver_iterations());
      
      table_convergence.add_value("XFEM-time",xmodel_simple.get_last_run_time());
      table_convergence.set_precision("XFEM-time", 3);
      }
      
      //write the table every cycle (to have at least some results if program fails)
      table_convergence.write_text(std::cout);
      std::ofstream out_file;
      out_file.open(output_dir + "table_convergence_circle.tex");
      table_convergence.write_tex(out_file);
      out_file.close();
    } 
    
  delete well;
  
  if(fine_triangulation != NULL)
    delete fine_triangulation;
  
  std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE - DONE ::::::::::::::::\n\n" << std::endl;
}
//*/

void test_convergence_square(std::string output_dir)
{
  std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE ::::::::::::::::\n\n" << std::endl;
  
  //------------------------------SETTING----------------------------------
  std::string test_name = "square_convergence_";
  bool fem_create = false,
       fem = false,
       xfem = true,
       ex = false;
  
  double p_a = 100.0,    //area of the model
         p_b = 100.0,
         excenter = 5.43,
         radius = p_a*std::sqrt(2),
         well_radius = 0.2,
         perm2fer = Parameters::perm2fer, 
         perm2tard = Parameters::perm2tard,
         transmisivity = Parameters::transmisivity,
         enrichment_radius = 50.0,
         well_pressure = 50*Parameters::pressure_at_top;
  
  unsigned int n_well_q_points = 200,
               initial_refinement = 2;
         
  Point<2> well_center(0+excenter,0+excenter);
  
  std::string input_dir = "../input/square_convergence/";
  std::string coarse_file = input_dir + "coarse_grid.msh";

  //--------------------------END SETTING----------------------------------
  
  Point<2> down_left(-p_a,-p_a);
  Point<2> up_right(p_a, p_a);
  std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
  
  
  Well *well = new Well( well_radius,
                         well_center);
//   well->set_perm2aquifer({perm2fer, perm2fer});
//   well->set_perm2aquitard({perm2tard, 0.0});
  well->set_perm2aquifer(0,perm2fer);
  well->set_perm2aquitard({perm2tard, 0.0});
  well->set_pressure(well_pressure);
  well->evaluate_q_points(n_well_q_points);
  
  //the radius is the half of the diagonal of the square: 2*p_a*sqrt(2)/2 = p_a*sqrt(2)
  Solution::ExactBase* exact_solution = new Solution::ExactSolution(well, radius);
  //Function<2> *dirichlet_square = new Solution::ExactSolution(well,radius);
  
  //FEM model creation
  Model_simple model_simple(well);  
  model_simple.set_name(test_name + "fem");
  model_simple.set_output_dir(output_dir);
  model_simple.set_area(down_left,up_right);
  model_simple.set_transmisivity(transmisivity,0);
  model_simple.set_initial_refinement(initial_refinement);  
  model_simple.set_ref_coarse_percentage(0.3,0.0);
  //model_simple.set_ref_coarse_percentage(0.3,0.05);
  //model_simple.set_grid_create_type(ModelBase::rect);
  
  model_simple.set_grid_create_type(ModelBase::rect);
  //model_simple.set_computational_mesh(coarse_file);
  model_simple.set_dirichlet_function(exact_solution);
  model_simple.set_adaptivity(true);
  model_simple.set_output_options(ModelBase::output_gmsh_mesh
                                | ModelBase::output_solution
                                | ModelBase::output_error);
  
  XModel_simple xmodel(well, ""); 
//     xmodel.set_name(test_name + "xfem");
//     xmodel.set_enrichment_method(Enrichment_method::xfem);
//     xmodel.set_name(test_name + "xfem_ramp");
//     xmodel.set_enrichment_method(Enrichment_method::xfem_ramp);
    xmodel.set_name(test_name + "xfem_shift");
    xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
//     xmodel.set_name(test_name + "sgfem"); 
//     xmodel.set_enrichment_method(Enrichment_method::sgfem);
  
  xmodel.set_output_dir(output_dir);
  xmodel.set_area(down_left,up_right);
  xmodel.set_transmisivity(transmisivity,0);
  xmodel.set_initial_refinement(initial_refinement);                                     
  xmodel.set_enrichment_radius(enrichment_radius);
  xmodel.set_grid_create_type(ModelBase::rect);
  xmodel.set_dirichlet_function(exact_solution);
  xmodel.set_adaptivity(true);
  //xmodel.set_well_computation_type(Well_computation::sources);
  xmodel.set_output_options(ModelBase::output_gmsh_mesh
                          //| ModelBase::output_solution
                          //| ModelBase::output_decomposed
                          //| ModelBase::output_adaptive_plot
                          | ModelBase::output_error);

//   // Exact model
//   if(ex)
//   {
//     std::cout << "computing exact solution on fine mesh..." << std::endl;
     
//     exact.output_distributed_solution(*fine_triangulation);
//     ExactBase* exact_solution = new ExactSolution(well, radius);
//     double exact_norm = Comparing::L2_norm_exact(*fine_triangulation,exact_solution);
//     std::cout << "L2_norm of the exact solution: " << exact_norm << std::endl;
//     //return;
//   }
  
  
  unsigned int n_cycles = 15;
  std::cout << "Cycles: " << n_cycles << std::endl;
  std::pair<double,double> l2_norm_dif_fem, l2_norm_dif_xfem;
  
  ConvergenceTable table_convergence;
  
    for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
        table_convergence.add_value("Cycle",cycle);
        table_convergence.set_tex_format("Cycle", "r");
      
        if(fem)
        {
            std::cout << "===== FEM Model_simple running   " << cycle << "   =====" << std::endl;
            model_simple.run (cycle);
            //model_simple.output_distributed_solution(*fine_triangulation, cycle);
            model_simple.output_results (cycle);
//       l2_norm_dif_fem = Comparing::L2_norm_diff( model_simple.get_distributed_solution(),
//                                                  *fine_triangulation,
//                                                  exact_solution);
      
            Vector<double> diff_vector;
            l2_norm_dif_fem = model_simple.integrate_difference(diff_vector, *exact_solution);
      
            table_convergence.add_value("L2",l2_norm_dif_fem.second);
            table_convergence.set_tex_caption("L2","$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$");
            table_convergence.set_precision("L2", 2);
            table_convergence.set_scientific("L2",true);
  
            table_convergence.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
            table_convergence.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
      
            table_convergence.add_value("FEM-dofs",model_simple.get_number_of_dofs());
            table_convergence.add_value("It_{FEM}",model_simple.solver_iterations());
      
            table_convergence.add_value("FEM-time",model_simple.last_run_time());
            table_convergence.set_precision("FEM-time", 3);

            table_convergence.set_tex_format("FEM-dofs", "r");
            table_convergence.set_tex_format("It_{FEM}", "r");
            table_convergence.set_tex_format("FEM-time", "r");
            std::cout << "===== FEM Model_simple finished =====" << std::endl;
        }
      
      
        if(xfem)
        {
            std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
            xmodel.run (cycle);  
      
            //xmodel.output_distributed_solution(*fine_triangulation,cycle);
            std::cout << "===== XModel_simple finished =====" << std::endl;
      
//       std::cout << "L2 norm of exact solution = " << Comparing::L2_norm_exact(xmodel.get_output_triangulation(), 
//                                                                               exact_solution) << std::endl;
      
//       l2_norm_dif_xfem = Comparing::L2_norm_diff( xmodel.get_distributed_solution(),
//                                                   xmodel.get_output_triangulation(),
//                                                   exact_solution);
  
      Vector<double> diff_vector;
      l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, *exact_solution);
      
      table_convergence.add_value("X_L2",l2_norm_dif_xfem.second);
      table_convergence.set_tex_caption("X_L2","$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$");
      table_convergence.set_precision("X_L2", 2);
      table_convergence.set_scientific("X_L2",true);
      
      table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate);
      table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate_log2);
      
//       table_convergence.add_value("X_L2 nodal",l2_norm_dif_xfem.first);
//       table_convergence.set_tex_caption("X_L2 nodal","$\\|x_{XFEM}-x_{exact}\\|_{L^2(N)}$");
//       table_convergence.set_precision("X_L2 nodal", 2);
//       table_convergence.set_scientific("X_L2 nodal",true);
      
//       table_convergence.evaluate_convergence_rates("X_L2 nodal", ConvergenceTable::reduction_rate);
//       table_convergence.evaluate_convergence_rates("X_L2 nodal", ConvergenceTable::reduction_rate_log2);
      
      table_convergence.add_value("XFEM-dofs",xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
      table_convergence.add_value("XFEM-enriched dofs",xmodel.get_number_of_dofs().second);
      table_convergence.add_value("It_{XFEM}",xmodel.solver_iterations());
      
      table_convergence.add_value("XFEM-time",xmodel.last_run_time());
      table_convergence.set_precision("XFEM-time", 3);
      
      table_convergence.set_tex_format("XFEM-dofs", "r");
      table_convergence.set_tex_format("XFEM-enriched dofs", "r");
      table_convergence.set_tex_format("It_{XFEM}", "r");
      table_convergence.set_tex_format("XFEM-time", "r");

       }
      
        //write the table every cycle (to have at least some results if program fails)
        table_convergence.write_text(std::cout);
        std::ofstream out_file;
        out_file.open(output_dir + xmodel.name() + ".tex");
        table_convergence.write_tex(out_file);
        out_file.close();
        
        out_file.open(output_dir + xmodel.name() + ".txt");
        table_convergence.write_text(out_file, 
                                     TableHandler::TextOutputFormat::table_with_separate_column_description);
        out_file.close();
      
        if(xfem)
        {
            xmodel.compute_interpolated_exact(exact_solution);
            xmodel.output_results(cycle);
//             ExactModel* exact = new ExactModel(exact_solution);
//             exact->output_distributed_solution(xmodel.get_output_triangulation(), cycle);
//             delete exact;
        } 
    }   // for cycle
      
  delete well;
  delete exact_solution;
  
  std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE - DONE ::::::::::::::::\n\n" << std::endl;
}


void test_convergence_sin(std::string output_dir)
{
  std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH SIN(x) ::::::::::::::::\n\n" << std::endl;
  
  //------------------------------SETTING----------------------------------
  std::string test_name = "sin_square_convergence_";
  bool fem_create = false,
       fem = false,
       xfem = true,
       ex = false;
  
  double p_a = 10.0,    //area of the model
         p_b = 10.0,
         excenter = 0.0, //0.05,
         //the radius is the half of the diagonal of the square: 2*p_a*sqrt(2)/2 = p_a*sqrt(2)
         radius = p_a*std::sqrt(2),
         well_radius = 0.02,
         perm2fer = 1e5,//Parameters::perm2fer, 
         perm2tard = Parameters::perm2tard,
         transmisivity = Parameters::transmisivity,
         enrichment_radius = 4.0,
         well_pressure = Parameters::pressure_at_top,
         k_wave_num = 0.3,
         amplitude = 0.8;
         
  unsigned int n_well_q_points = 200;
         
  Point<2> well_center(0,0);
  
  std::string input_dir = "../input/square_convergence/";
  std::string coarse_file = input_dir + "coarse_grid.msh";

  //--------------------------END SETTING----------------------------------
  
  Point<2> down_left(-p_a+excenter,-p_a+excenter);
  Point<2> up_right(p_a+excenter, p_a+excenter);
  std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
  
  
  Well *well = new Well( well_radius,
                         well_center);
  well->set_pressure(well_pressure);
  well->set_perm2aquifer(0,perm2fer);
  well->set_perm2aquitard({perm2tard, 0.0});
  well->evaluate_q_points(n_well_q_points);
  
  ExactSolution1 *exact_solution = new Solution::ExactSolution1(well, radius, k_wave_num, amplitude);
  Function<2> *dirichlet_square = exact_solution;
  Function<2> *rhs_function = new Solution::Source1(*exact_solution);
  
  //FEM model creation
  Model_simple model(well);  
  model.set_name(test_name + "fem");
  model.set_output_dir(output_dir);
  model.set_area(down_left,up_right);
  model.set_transmisivity(transmisivity,0);
  model.set_initial_refinement(2);  
  model.set_ref_coarse_percentage(0.3,0.0);
  //model.set_grid_create_type(ModelBase::rect);
  
  model.set_grid_create_type(ModelBase::rect);
  //model.set_computational_mesh(coarse_file);
  model.set_dirichlet_function(dirichlet_square);
  model.set_rhs_function(rhs_function);
  model.set_adaptivity(true);
  model.set_output_options(ModelBase::output_gmsh_mesh
                         | ModelBase::output_solution
                         | ModelBase::output_error);
    
  
  XModel_simple xmodel(well);  
//   xmodel.set_name(test_name + "sgfem");
//   xmodel.set_enrichment_method(Enrichment_method::sgfem);
  xmodel.set_name(test_name + "xfem_shift");
  xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
//   xmodel.set_name(test_name + "xfem_ramp");
//   xmodel.set_enrichment_method(Enrichment_method::xfem_ramp);
  
  xmodel.set_output_dir(output_dir);
  xmodel.set_area(down_left,up_right);
  xmodel.set_transmisivity(transmisivity,0);
  xmodel.set_initial_refinement(3);                                     
  xmodel.set_enrichment_radius(enrichment_radius);
  xmodel.set_grid_create_type(ModelBase::rect);
  xmodel.set_dirichlet_function(dirichlet_square);
  xmodel.set_rhs_function(rhs_function);
  xmodel.set_adaptivity(true);
  xmodel.set_output_options(ModelBase::output_gmsh_mesh
                          | ModelBase::output_solution
                          | ModelBase::output_decomposed
                          | ModelBase::output_error);
  
     ExactModel exact(exact_solution);

  unsigned int n_cycles = 15;
  std::pair<double,double> l2_norm_dif_fem, l2_norm_dif_xfem;
  
  ConvergenceTable table_convergence;
  
    for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
        table_convergence.add_value("Cycle",cycle);
        table_convergence.set_tex_format("Cycle", "r");
        if(fem)
        {
            std::cout << "===== FEM Model_simple running   " << cycle << "   =====" << std::endl;
            model.run (cycle);
//       model.output_distributed_solution(*fine_triangulation, cycle);
            model.output_results (cycle);
//       l2_norm_dif_fem = Comparing::L2_norm_diff( model.get_distributed_solution(),
//                                                  *fine_triangulation,
//                                                  exact_solution);
      
            Vector<double> diff_vector;
            l2_norm_dif_fem = model.integrate_difference(diff_vector, *exact_solution);
      
            table_convergence.add_value("L2",l2_norm_dif_fem.second);
            table_convergence.set_tex_caption("L2","$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$");
            table_convergence.set_precision("L2", 2);
            table_convergence.set_scientific("L2",true);
  
            table_convergence.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
            table_convergence.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
      
            table_convergence.add_value("FEM-dofs",model.get_number_of_dofs());
            table_convergence.add_value("It_{FEM}",model.solver_iterations());
      
            table_convergence.add_value("FEM-time",model.last_run_time());
            table_convergence.set_precision("FEM-time", 3);

            table_convergence.set_tex_format("FEM-dofs", "r");
            table_convergence.set_tex_format("It_{FEM}", "r");
            table_convergence.set_tex_format("FEM-time", "r");
            std::cout << "===== FEM Model_simple finished =====" << std::endl;
        }
      
      
        if(xfem)
        {
            std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
            xmodel.run (cycle);  
      
            //xmodel.output_distributed_solution(*fine_triangulation,cycle);
            std::cout << "===== XModel_simple finished =====" << std::endl;
      
//       std::cout << "L2 norm of exact solution = " << Comparing::L2_norm_exact(xmodel.get_output_triangulation(), 
//                                                                               exact_solution) << std::endl;
      
//       l2_norm_dif_xfem = Comparing::L2_norm_diff( xmodel.get_distributed_solution(),
//                                                   xmodel.get_output_triangulation(),
//                                                   exact_solution);
      Vector<double> diff_vector;
      l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, *exact_solution);
      
      table_convergence.add_value("X_L2",l2_norm_dif_xfem.second);
      table_convergence.set_tex_caption("X_L2","$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$");
      table_convergence.set_precision("X_L2", 2);
      table_convergence.set_scientific("X_L2",true);
      
      table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate);
      table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate_log2);
      
//       table_convergence.add_value("X_L2 nodal",l2_norm_dif_xfem.first);
//       table_convergence.set_tex_caption("X_L2 nodal","$\\|x_{XFEM}-x_{exact}\\|_{L^2(N)}$");
//       table_convergence.set_precision("X_L2 nodal", 2);
//       table_convergence.set_scientific("X_L2 nodal",true);
//       
//       table_convergence.evaluate_convergence_rates("X_L2 nodal", ConvergenceTable::reduction_rate);
//       table_convergence.evaluate_convergence_rates("X_L2 nodal", ConvergenceTable::reduction_rate_log2);
      
      table_convergence.add_value("XFEM-dofs",xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
      table_convergence.add_value("XFEM-enriched dofs",xmodel.get_number_of_dofs().second);
      table_convergence.add_value("It_{XFEM}",xmodel.solver_iterations());
      
      table_convergence.add_value("XFEM-time",xmodel.last_run_time());
      table_convergence.set_precision("XFEM-time", 3);
      
      table_convergence.set_tex_format("XFEM-dofs", "r");
      table_convergence.set_tex_format("XFEM-enriched dofs", "r");
      table_convergence.set_tex_format("It_{XFEM}", "r");
      table_convergence.set_tex_format("XFEM-time", "r");
      }
      
      //write the table every cycle (to have at least some results if program fails)
      table_convergence.write_text(std::cout);
      std::ofstream out_file;
      out_file.open(output_dir + xmodel.name() + ".tex");
      table_convergence.write_tex(out_file);
      out_file.close();
      
      out_file.open(output_dir + xmodel.name() + ".txt");
      table_convergence.write_text(out_file, 
                                   TableHandler::TextOutputFormat::table_with_separate_column_description);
      out_file.close();
      
      if(xfem)
      {
        xmodel.compute_interpolated_exact(exact_solution);
        xmodel.output_results(cycle);
        exact.output_distributed_solution(xmodel.get_output_triangulation(), cycle);
      }
    } 
    
    delete well;
    delete exact_solution;
    delete rhs_function;
  
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH SIN(x) - DONE ::::::::::::::::\n\n" << std::endl;
}


void test_multiple_wells(std::string output_dir)
{
  std::cout << "\n\n:::::::::::::::: MULTIPLE WELLS TEST ::::::::::::::::\n\n" << std::endl;
  
  //------------------------------SETTING----------------------------------
  std::string test_name = "multiple_";
  bool fem_create = false;
  
  double p_a = 15.0,    //area of the model
         p_b = 15.0,
         well_radius = 0.02,
         perm2fer = Parameters::perm2fer, 
         perm2tard = Parameters::perm2tard,
         transmisivity = Parameters::transmisivity,
         enrichment_radius = 3.0;
         
  unsigned int n_well_q_points = 500;
         
  
  //std::string input_dir = "../input/square_convergence/";
  std::string input_dir = "../input/multiple/";
  std::string coarse_file = input_dir + "real_grid_7.msh";
  
  //std::string coarse_file = input_dir + "coarse_grid.msh";
  std::string ref_flags_fine = input_dir + "ref_flags_7.ptf";
  
  //PersistentTriangulation<2>* fine_triangulation = NULL;
  //--------------------------END SETTING----------------------------------
  
  Point<2> down_left(-p_a,-p_a);
  Point<2> up_right(p_a, p_a);
  std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
  
  
  //vector of wells
  std::vector<Well*> wells;
  
  wells.push_back( new Well( Parameters::radius,
                             Point<2>(-10.0,-10.0),
                             Parameters::perm2fer, 
                             Parameters::perm2tard));
  
  wells.push_back( new Well( Parameters::radius,
                             Point<2>(0.0,0.0), 
                             Parameters::perm2fer, 
                             Parameters::perm2tard));
    
  wells.push_back( new Well( Parameters::radius,
                             Point<2>(-10.0,8.0), 
                             Parameters::perm2fer, 
                             Parameters::perm2tard));
    
  wells.push_back( new Well( Parameters::radius,
                             Point<2>(8.0,9.0), 
                             Parameters::perm2fer, 
                             Parameters::perm2tard));
    
  wells.push_back( new Well( Parameters::radius,
                             Point<2>(5.0,-10.0), 
                             Parameters::perm2fer, 
                             Parameters::perm2tard));
    
  //setting BC - pressure at the top of the wells
  wells[0]->set_pressure(-2.5*Parameters::pressure_at_top);
  wells[1]->set_pressure(-2*Parameters::pressure_at_top);
  wells[2]->set_pressure(2*Parameters::pressure_at_top);
  wells[3]->set_pressure(Parameters::pressure_at_top);
  wells[4]->set_pressure(3*Parameters::pressure_at_top);
  
  for(unsigned int w=0; w < wells.size(); w++)
  {
    wells[w]->evaluate_q_points(n_well_q_points);
  }
  
  //the radius is the half of the diagonal of the square: 2*p_a*sqrt(2)/2 = p_a*sqrt(2)
  //Function<2> *dirichlet = new ZeroFunction<2>();
  
  //FEM model creation
  Model model_fem(wells);  
  model_fem.set_name(test_name + "fem");
  model_fem.set_output_dir(output_dir);
  model_fem.set_transmisivity(transmisivity,0); 
  //model_fem.set_dirichlet_function(dirichlet);
  
  XModel xmodel(wells);  
//   xmodel.set_name(test_name + "sgfem");
//   xmodel.set_enrichment_method(Enrichment_method::sgfem);
  xmodel.set_name(test_name + "xfem_shift");
  xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
  
  xmodel.set_output_dir(output_dir);
  xmodel.set_area(down_left,up_right);
  xmodel.set_transmisivity(transmisivity,0);
  xmodel.set_initial_refinement(3);                                     
  xmodel.set_enrichment_radius(enrichment_radius);
  xmodel.set_grid_create_type(ModelBase::rect);
  //xmodel.set_dirichlet_function(dirichlet);
  xmodel.set_adaptivity(true);
  //xmodel.set_well_computation_type(Well_computation::sources);

  
  
  if(fem_create)
  {
    model_fem.set_area(down_left,up_right);
    model_fem.set_initial_refinement(5); 
    model_fem.set_grid_create_type(ModelBase::rect);
    model_fem.set_ref_coarse_percentage(0.3,0.05);
    model_fem.set_adaptivity(true);
    for (unsigned int cycle=0; cycle < 15; ++cycle)
    { 
      std::cout << "===== Model running   " << cycle << "   =====" << std::endl;
      model_fem.run (cycle);
      model_fem.output_results (cycle);
      std::cout << "===== Model finished =====" << std::endl;
    }
    return;
  }
  
  
  model_fem.set_grid_create_type(ModelBase::load);
  //model_fem.set_computational_mesh(coarse_file, ref_flags_fine);
  model_fem.set_computational_mesh(coarse_file);
  model_fem.run();
  double l2_norm_fem = Comparing::L2_norm( model_fem.get_solution(),
                                           model_fem.get_triangulation()
                                          );
  
  std::cout << "l2 norm of fem solution: "  << l2_norm_fem << std::endl;
  return;
  
  unsigned int n_cycles = 4;
  double l2_norm_dif;
  
  TableHandler table;
  
  for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
      table.add_value("Cycle",cycle);
 
      std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
      xmodel.run (cycle);  
      xmodel.output_distributed_solution(model_fem.get_triangulation(),cycle);
      std::cout << "===== XModel_simple finished =====" << std::endl;
      
      l2_norm_dif = Comparing::L2_norm_diff( model_fem.get_solution(),
                                             xmodel.get_distributed_solution(),
                                             model_fem.get_triangulation()
                                           );
      
      table.add_value("$\\|x_{XFEM}-x_{FEM}\\|_{L^2(\\Omega)}$",l2_norm_dif);
      table.set_precision("$\\|x_{XFEM}-x_{FEM}\\|_{L^2(\\Omega)}$", 2);
      table.set_scientific("$\\|x_{XFEM}-x_{FEM}\\|_{L^2(\\Omega)}$",true);
      
      table.add_value("dofs",xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
      table.add_value("enriched dofs",xmodel.get_number_of_dofs().second);
      table.add_value("Iterations",xmodel.solver_iterations());
      
      //table.add_value("XFEM-time",xmodel.get_last_run_time());
      //table.set_precision("XFEM-time", 3);

      
      //write the table every cycle (to have at least some results if program fails)
      table.write_text(std::cout);
      std::ofstream out_file;
      out_file.open(output_dir + xmodel.name() + ".tex");
      table.write_tex(out_file);
      out_file.close();
    } 
  //*/
  std::cout << "\n\n:::::::::::::::: MULTIPLE WELLS TEST END ::::::::::::::::\n\n" << std::endl;
}
  

void test_output(std::string output_dir)
{
  std::string test_name = "test_output_";
  double p_a = 10.0,    //area of the model
         p_b = 10.0,
         radius = p_a*std::sqrt(2),
         well_radius = 0.02,
         perm2fer = Parameters::perm2fer, 
         perm2tard = Parameters::perm2tard,
         transmisivity = Parameters::transmisivity,
         enrichment_radius = 2.0;
         
  unsigned int n_well_q_points = 500;
         
  Point<2> well_center(0,0);
  
  //std::string input_dir = "../input/square_convergence/";
  //std::string coarse_file = input_dir + "coarse_grid.msh";
  //std::string coarse_file_0 = input_dir + "coarse_grid.msh";
  //std::string ref_flags_coarse = input_dir + "ref_flags_1.ptf";
  //std::string ref_flags_fine = input_dir + "ref_flags_7.ptf";
  
  //--------------------------END SETTING----------------------------------
  
  Point<2> down_left(-p_a,-p_a);
  Point<2> up_right(p_a, p_a);
  std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
  
  
  Well *well = new Well( well_radius,
                         well_center,
                         perm2fer, 
                         perm2tard);
  well->set_pressure(Parameters::pressure_at_top);
  well->evaluate_q_points(n_well_q_points);
  
  //the radius is the half of the diagonal of the square: 2*p_a*sqrt(2)/2 = p_a*sqrt(2)
  Function<2> *dirichlet = new ExactSolution(well,radius);
  
  XModel_simple xmodel(well);  
  
  //xmodel.set_name(test_name + "sgfem_new");
  //xmodel.set_name(test_name + "sgfem_oldnew");
//  xmodel.set_name(test_name + "sgfem");
//  xmodel.set_enrichment_method(Enrichment_method::sgfem);
  
  //xmodel.set_name(test_name + "xfem_shift_new");
  //xmodel.set_name(test_name + "xfem_shift_oldnew");
  xmodel.set_name(test_name + "xfem_shift");
  xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
  
  xmodel.set_output_dir(output_dir);
  xmodel.set_area(down_left,up_right);
  xmodel.set_transmisivity(transmisivity,0);
  xmodel.set_initial_refinement(4);                                     
  xmodel.set_enrichment_radius(enrichment_radius);
  xmodel.set_grid_create_type(ModelBase::rect);
  xmodel.set_dirichlet_function(dirichlet);
  //xmodel.set_adaptivity(true);
  //xmodel.set_well_computation_type(Well_computation::sources);
  
  xmodel.run();
  xmodel.output_results();
  
  ExactBase* exact_solution = new ExactSolution(well, radius);
  double l2_norm_dif_xfem = Comparing::L2_norm_diff(xmodel.get_distributed_solution(),
                                                    xmodel.get_output_triangulation(),
                                                    exact_solution);
  std::cout << "l2_norm of difference to exact solution: " << l2_norm_dif_xfem << std::endl;
}


void test_solution(std::string output_dir)
{
    std::cout << "\n\n:::::::::::::::: TEST SOLUTION ON SQUARE ::::::::::::::::\n\n" << std::endl;
  
  //------------------------------SETTING----------------------------------
  std::string test_name = "test_solution_";
  
  double p_a = 10.0,    //area of the model
         p_b = 10.0,
         excenter = 0,//0.06, //0.05,
         radius = p_a*std::sqrt(2),
         well_radius = 0.02,
         perm2fer = Parameters::perm2fer, 
         perm2tard = Parameters::perm2tard,
         transmisivity = Parameters::transmisivity,
         enrichment_radius = 15.0,
         well_pressure = Parameters::pressure_at_top;
         
  unsigned int n_well_q_points = 200;
         
  Point<2> well_center(0+excenter,0+excenter);

  //--------------------------END SETTING----------------------------------
  
  Point<2> down_left(-p_a,-p_a);
  Point<2> up_right(p_a, p_a);
  std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
  
  
  Well *well = new Well( well_radius,
                         well_center,
                         perm2fer, 
                         perm2tard);
  well->set_pressure(well_pressure);
  well->evaluate_q_points(n_well_q_points);
  
  //the radius is the half of the diagonal of the square: 2*p_a*sqrt(2)/2 = p_a*sqrt(2)
  Solution::ExactBase* exact_solution = new Solution::ExactSolution(well, radius);
  //Function<2> *dirichlet_square = new Solution::ExactSolution(well,radius);
  
  XModel_simple xmodel(well);  
//   xmodel.set_name(test_name + "sgfem_model"); 
//   xmodel.set_enrichment_method(Enrichment_method::sgfem);
  xmodel.set_name(test_name + "xfem_shift_model");
  xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
  
  xmodel.set_output_dir(output_dir);
  xmodel.set_area(down_left,up_right);
  xmodel.set_transmisivity(transmisivity,0);
  xmodel.set_initial_refinement(3);                                     
  xmodel.set_enrichment_radius(enrichment_radius);
  xmodel.set_grid_create_type(ModelBase::rect);
  xmodel.set_dirichlet_function(exact_solution);
  xmodel.set_adaptivity(true);
  //xmodel.set_well_computation_type(Well_computation::sources);
  xmodel.set_output_options(ModelBase::output_gmsh_mesh
                          | ModelBase::output_solution
                          | ModelBase::output_decomposed
                          | ModelBase::output_error);
  
//   // Exact model
//   if(ex)
//   {
//     std::cout << "computing exact solution on fine mesh..." << std::endl;
     ExactModel* exact = new ExactModel(exact_solution);
//     exact.output_distributed_solution(*fine_triangulation);
//     ExactBase* exact_solution = new ExactSolution(well, radius);
//     double exact_norm = Comparing::L2_norm_exact(*fine_triangulation,exact_solution);
//     std::cout << "L2_norm of the exact solution: " << exact_norm << std::endl;
//     //return;
//   }
  
  
  double l2_norm_dif_fem;
  std::pair<double,double> l2_norm_dif_xfem;
 
      std::cout << "===== XModel_simple running   =====" << std::endl;
      
      xmodel.test_method(exact_solution);  
      
      
      std::cout << "===== XModel_simple finished =====" << std::endl;
      

      Vector<double> diff_vector;
      l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, *exact_solution);
      
      xmodel.compute_interpolated_exact(exact_solution);
      xmodel.output_results();
      exact->output_distributed_solution(xmodel.get_output_triangulation());

  delete exact;
  delete well;
  delete exact_solution;
  
  std::cout << "\n\n:::::::::::::::: TEST SOLUTION ON SQUARE - DONE ::::::::::::::::\n\n" << std::endl;
}


void test_adaptive_integration(std::string output_dir)
{
  output_dir += "test_adaptive_integration/";
  std::string test_name = "test_integration_";
  double p_a = 2.0,    //area of the model
         well_radius = 0.02,
         excenter = 0,//0.61,
         perm2fer = Parameters::perm2fer, 
         perm2tard = Parameters::perm2tard;
         
  unsigned int n_well_q_points = 500;
         
  Point<2> well_center(0+excenter,0+excenter);
  
  //--------------------------END SETTING----------------------------------
  
  Point<2> down_left(-p_a,-p_a);
  Point<2> up_right(p_a, p_a);
  std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
  
  
  Well *well = new Well( well_radius,
                         well_center,
                         perm2fer, 
                         perm2tard);
  well->set_pressure(well_radius);
  well->evaluate_q_points(n_well_q_points);
  
//   Well *well2 = new Well( 1.0,
//                          well_center,
//                          perm2fer, 
//                          perm2tard);
//   well2->set_pressure(1.0);
//   well2->evaluate_q_points(n_well_q_points);
  
  Triangulation<2> tria;
  GridGenerator::hyper_rectangle<2>(tria,down_left, up_right);
  DBGMSG("tria size: %d\n",tria.n_active_cells());
  
  FE_Q<2> fe(1);
  QGauss<2> quad(3);
  FEValues<2> fe_values(fe, quad, UpdateFlags::update_default);
  DoFHandler<2> dof_handler;
  dof_handler.initialize(tria, fe);
  
  DoFHandler<2>::active_cell_iterator cell = dof_handler.begin_active();  
  
  std::vector<unsigned int> enriched_dofs(fe.dofs_per_cell);
  enriched_dofs[0] = 4;
  enriched_dofs[1] = 5;
  enriched_dofs[2] = 6;
  enriched_dofs[3] = 7;
  std::vector<unsigned int> weights(fe.dofs_per_cell,0);
  
  std::vector<const Point<2>* > points (well->q_points().size());
  //std::vector<const Point<2>* > points2 (well2->q_points().size());
  for(unsigned int p=0; p < well->q_points().size(); p++)
  {
        points[p] = &(well->q_points()[p]);
        //points2[p] = &(well2->q_points()[p]);
  }
  DBGMSG("N quadrature points: %d\n",points.size());
  XDataCell* xdata = new XDataCell(cell, well,0,enriched_dofs, weights, points);
  //xdata->add_data(well,1,enriched_dofs,weights,points2);
  
  cell->set_user_pointer(xdata);
  fe_values.reinit(cell);

  Adaptive_integration adapt(cell, fe,fe_values.get_mapping(),0);
  
  for(unsigned int i=0; i < 10; i++)
    adapt.refine_edge();
  
  adapt.gnuplot_refinement(output_dir,true, true);
  
  TestIntegration* func = new TestIntegration(well);
  
  std::cout << setprecision(16) << adapt.test_integration(func) << std::endl;
  
  //std::cout << func->value(Point<2>(0,0)) << "  " << std::log(0.5) << "  " << func->value(Point<2>(0.5,0)) << std::endl;
  //x^2
//   double integral = well_radius * log(well_radius);
//   integral += p_a*log(p_a) - p_a - (well_radius*log(well_radius) - well_radius);
//   integral = 2*4*integral;
//   std::cout << setprecision(16) << integral << std::endl;
  double integral = 0;//well_radius * well_radius * log(well_radius) * M_PI;
  integral += M_PI * (-0.5 - well_radius*well_radius*(std::log(well_radius)-0.5));
  std::cout << setprecision(16) << integral << std::endl;
}


void test_adaptive_integration2(std::string output_dir)
{
    output_dir += "test_adaptive_integration_2/";
  
    unsigned int start_level = 5,
               fine_level = 12;
               
    double p_a = 2.0,    //area of the model
           well_radius = 1.0,
           excenter = 0;
         
    Point<2> well_center(0+excenter,0+excenter);
  
    //--------------------------END SETTING----------------------------------
  
    Point<2> down_left(-p_a,-p_a);
    Point<2> up_right(p_a, p_a);
    std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
  
  
    Well *well = new Well( well_radius,
                           well_center);
    well->set_pressure(1.0);
    well->set_perm2aquifer(0,0);
    well->set_perm2aquitard({0,0});
    well->evaluate_q_points(100);
    
    Triangulation<2> tria;
    GridGenerator::hyper_rectangle<2>(tria,down_left, up_right);
    DBGMSG("tria size: %d\n",tria.n_active_cells());
    
    FE_Q<2> fe(1);
    QGauss<2> quad(1);
    FEValues<2> fe_values(fe, quad, UpdateFlags::update_default);
    DoFHandler<2> dof_handler;
    dof_handler.initialize(tria, fe);
    
    DoFHandler<2>::active_cell_iterator cell = dof_handler.begin_active();  
    
    std::vector<unsigned int> enriched_dofs(fe.dofs_per_cell);
    enriched_dofs[0] = 4;
    enriched_dofs[1] = 5;
    enriched_dofs[2] = 6;
    enriched_dofs[3] = 7;
    std::vector<unsigned int> weights(fe.dofs_per_cell,0);
    
    std::vector<const Point<2>* > points (well->q_points().size());
    //std::vector<const Point<2>* > points2 (well2->q_points().size());
    for(unsigned int p=0; p < well->q_points().size(); p++)
    {
            points[p] = &(well->q_points()[p]);
            //points2[p] = &(well2->q_points()[p]);
    }
    DBGMSG("N quadrature points: %d\n",points.size());
    XDataCell* xdata = new XDataCell(cell, well,0,enriched_dofs, weights, points);
    //xdata->add_data(well,1,enriched_dofs,weights,points2);
    
    cell->set_user_pointer(xdata);
    fe_values.reinit(cell);

    TestIntegration_r2* func = new TestIntegration_r2(well);
    
    ConvergenceTable table_convergence;
            
    for(unsigned int i=0; i < fine_level-start_level-1; i++)
    {
        Adaptive_integration adapt(cell, fe,fe_values.get_mapping(),0);
           
        // compute integral of the function on the well edge
        for(unsigned int j=0; j < start_level+i; j++)
            adapt.refine_edge();
  
    //     adapt.gnuplot_refinement(output_dir,true, true);
        // test, fine
        std::pair<double, double> integrals = adapt.test_integration_2(func, fine_level-start_level-i);
    
        double difference = integrals.second-integrals.first;
        std::cout << setprecision(16) << difference << std::endl;
        
        table_convergence.add_value("Refinement level",start_level+i);
        table_convergence.set_tex_format("Refinement level", "r");
      
        table_convergence.add_value("difference",difference);
        table_convergence.set_precision("difference", 3);
        table_convergence.set_scientific("difference",true);
  
        table_convergence.evaluate_convergence_rates("difference", ConvergenceTable::reduction_rate);
        table_convergence.evaluate_convergence_rates("difference", ConvergenceTable::reduction_rate_log2);
        table_convergence.write_text(std::cout);
    }
    table_convergence.write_text(std::cout);
    std::ofstream out_file;
    out_file.open(output_dir + "convergence.tex");
    table_convergence.write_tex(out_file);
    out_file.close();
        
    out_file.open(output_dir + "convergence.txt");
    table_convergence.write_text(out_file, 
                                 TableHandler::TextOutputFormat::table_with_separate_column_description);
    out_file.close();
}


void test_adaptive_integration3(std::string output_dir)
{
    std::string test_name = "test_adaptive_integration2_";
    std::string output_path;
    // test setting:
    const unsigned int refinement_level_min = 2,
                       refinement_level_max = 12,
                       n_refinement_levels = refinement_level_max - refinement_level_min + 1,
                       n_center_generations = 20;
                 
    double p_a = 2.0,    //area of the model
           well_radius = 1.0,
           perm2fer = Parameters::perm2fer, 
           perm2tard = Parameters::perm2tard;
         
    const unsigned int n_well_q_points = 500;
  
    std::vector<Point<2> > well_centers(n_center_generations);
    std::vector<Well*> wells(n_center_generations);
    
    std::vector<double> average_relative_errors(n_refinement_levels, 0);
    std::vector<double> max_relative_errors(n_refinement_levels, 0);
  
    WellCharacteristicFunction *well_characteristic_function;
    //--------------------------END SETTING----------------------------------
  
    Point<2> down_left(-p_a,-p_a);
    Point<2> up_right(p_a, p_a);
    std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
  
    //random center generation:
    srand(time(nullptr)); //set random seed

    for(unsigned int i=0; i < n_center_generations; i++)
    {
        double x = (-1.0) * rand() / RAND_MAX;   //<-1,0>
        double y = (-1.0) * x + (1.0+x) * rand() / RAND_MAX;   //<-x,1>
        DBGMSG("well center: %f  %f\n",x,y);  
        
        well_centers[i] = Point<2>(x,y);
        wells[i] = new Well(well_radius,
                            well_centers[i],
                            perm2fer, 
                            perm2tard);
        wells[i]->set_pressure(well_radius);
        wells[i]->evaluate_q_points(n_well_q_points);
    }
   
   
    for(unsigned int i=0; i < n_center_generations; i++)
    {
        well_characteristic_function = new WellCharacteristicFunction(wells[i]);
  
        XModel_simple xmodel(wells[i]);
        xmodel.set_name(test_name);
        xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
  
        xmodel.set_output_dir(output_dir);
        output_path = xmodel.output_dir();
        xmodel.set_area(down_left,up_right);
  
        xmodel.set_transmisivity(1.0,0);
        xmodel.set_initial_refinement(2);                                     
        xmodel.set_enrichment_radius(4*p_a);
        xmodel.set_grid_create_type(ModelBase::rect);
//   xmodel.set_dirichlet_function(exact_solution);
//   xmodel.set_adaptivity(true);
//   xmodel.set_well_computation_type(Well_computation::sources);
//   xmodel.set_output_options(ModelBase::output_gmsh_mesh
//                           | ModelBase::output_solution
//                           | ModelBase::output_decomposed
//                           | ModelBase::output_error);
        
        TableHandler table;
        for(unsigned int l=refinement_level_min; l <= refinement_level_max; l++ )
        {
            unsigned int index = l - refinement_level_min;
            double rel_error = xmodel.test_adaptive_integration(well_characteristic_function,l);
            average_relative_errors[index] += rel_error;
            max_relative_errors[index] = std::max(max_relative_errors[index], rel_error);
            
            table.add_value("level",l);
            table.add_value("rel_error", rel_error);
            table.set_precision("rel_error",6);
            table.set_scientific("rel_error",true);
        }
        // write the table in different streams
        table.write_text(std::cout);
        std::stringstream table_filename;
        table_filename << output_path << "table_" << i;
        std::ofstream out_file;
        out_file.open(table_filename.str() + ".tex");
        table.write_tex(out_file);
        out_file.close();
        out_file.clear();
        out_file.open(table_filename.str() + ".txt");
        table.write_text(out_file, 
                         TableHandler::TextOutputFormat::table_with_separate_column_description);
        out_file.close();
       
        delete well_characteristic_function;
    }
    
    TableHandler final_table;
    //averaging relative errors
    for(unsigned int l=refinement_level_min; l <= refinement_level_max; l++ )
    {
        unsigned int index = l - refinement_level_min;
        final_table.add_value("level",l);
        
        final_table.add_value("max_rel_error", max_relative_errors[index]);
        final_table.set_precision("max_rel_error",3);
        final_table.set_scientific("max_rel_error",true);
        
        average_relative_errors[index] /= n_center_generations;
        final_table.add_value("avg_rel_error", average_relative_errors[index]);
        final_table.set_precision("avg_rel_error",3);
        final_table.set_scientific("avg_rel_error",true);
    }
    // write the table in different streams
    final_table.write_text(cout);
    std::ofstream out_file;
    out_file.open(output_path + "final_table.tex");
    final_table.write_tex(out_file);
    out_file.close();
    out_file.clear();
    out_file.open(output_path + "final_table.txt");
    final_table.write_text(out_file, 
                           TableHandler::TextOutputFormat::table_with_separate_column_description);
    out_file.close();
    
    // output well centers
    std::ofstream well_centers_file;
    well_centers_file.open (output_path + "well_centers.dat");
    for(unsigned int i=0; i < n_center_generations; i++)
    {
        if (well_centers_file.is_open()) well_centers_file << well_centers[i] << "\n";
    }
    well_centers_file.close();
    
    for(unsigned int i=0; i < n_center_generations; i++)
    {
        delete wells[i];
    }
  //*/
}


void test_two_aquifers(std::string output_dir)
{
  std::cout << "\n\n:::::::::::::::: TWO AQUIFERS TEST ::::::::::::::::\n\n" << std::endl;
  
  //------------------------------SETTING----------------------------------
  std::string test_name = "two_aquifers_";
  
  double p_a = 10.0,    //area of the model
         p_b = 10.0,
         well_radius = 0.02,
         enrichment_radius = 2.0,
         input_pressure = 5;
         
  unsigned int n_aquifers = 2, 
               initial_refinement = 3,
               n_well_q_points = 200;
  
  std::vector<double> transmisivity_vec = {0.1, 0.1};
  // top / aq1 / aq2
  std::vector<double> perm2fer_1 = {1e6, 0.0, 1e6};
  std::vector<double> perm2tard_1 = {1e10, 1e10, 0.0};
  std::vector<double> perm2fer_2 = {0.0, 1e6, 1e6};
  std::vector<double> perm2tard_2 = {1e10, 1e10, 0.0};
  
  //--------------------------END SETTING----------------------------------
  
  Point<2> down_left(-p_a,-p_a);
  Point<2> up_right(p_a, p_a);
  std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
  
  
  //vector of wells
  std::vector<Well*> wells;
  
  wells.push_back( new Well( Parameters::radius,
                             Point<2>(-5.0,-1.1)));
  wells[0]->set_pressure(input_pressure);
  wells[0]->set_perm2aquifer(perm2fer_1);
  wells[0]->set_perm2aquitard(perm2tard_1);
  
  wells.push_back( new Well( Parameters::radius,
                             Point<2>(5.0,1.0)));
  //wells[1]->set_pressure(0.0);
  wells[1]->set_perm2aquifer(perm2fer_2);
  wells[1]->set_perm2aquitard(perm2tard_2);  
  
  for(unsigned int w=0; w < wells.size(); w++)
  {
    wells[w]->evaluate_q_points(n_well_q_points);
  }
  
  
  XModel xmodel(wells, "",n_aquifers);  
  xmodel.set_name(test_name + "sgfem");
  xmodel.set_enrichment_method(Enrichment_method::sgfem);
//   xmodel.set_name(test_name + "xfem_shift");
//   xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
  
  xmodel.set_output_dir(output_dir);
  xmodel.set_area(down_left,up_right);
  xmodel.set_transmisivity(transmisivity_vec);
  xmodel.set_initial_refinement(initial_refinement);                                      
  xmodel.set_enrichment_radius(enrichment_radius);
  xmodel.set_grid_create_type(ModelBase::rect);
  //xmodel.set_dirichlet_function(dirichlet);
  xmodel.set_adaptivity(true);
  xmodel.set_output_options(ModelBase::output_gmsh_mesh
                          | ModelBase::output_solution
                          | ModelBase::output_decomposed
                          //| ModelBase::output_adaptive_plot
                          //| ModelBase::output_error
                           );
  
  unsigned int n_cycles = 1;
  
  TableHandler table;
  
  for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
      table.add_value("Cycle",cycle);
 
      std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
      xmodel.run (cycle);  
      
      std::cout << "===== XModel_simple finished =====" << std::endl;
      
      table.add_value("dofs",xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
      table.add_value("enriched dofs",xmodel.get_number_of_dofs().second);
      table.add_value("Iterations",xmodel.solver_iterations());
      
      //table.add_value("XFEM-time",xmodel.get_last_run_time());
      //table.set_precision("XFEM-time", 3);

      
      //write the table every cycle (to have at least some results if program fails)
      table.write_text(std::cout);
      xmodel.output_results(cycle);
//       std::ofstream out_file;
//       out_file.open(output_dir + xmodel.name() + ".tex");
//       table.write_tex(out_file);
//       out_file.close();
    } 
  //*/
  std::cout << "\n\n:::::::::::::::: MULTIPLE WELLS TEST END ::::::::::::::::\n\n" << std::endl;
  
}


void test_enr_error(std::string output_dir)
{
    std::cout << "\n\n:::::::::::::::: TEST ENRICHMENT RADIUS ERROR ::::::::::::::::\n\n" << std::endl;
    
  //------------------------------SETTING----------------------------------
  double p_a = 10.0,    //area of the model
         p_b = 10.0,
         excenter = 0,
         radius = p_a*std::sqrt(2),
         well_radius = 0.02,
         perm2fer = Parameters::perm2fer, 
         perm2tard = Parameters::perm2tard,
         transmisivity = 1.0,
         enrichment_radius = 2.0,
         well_pressure = 1.0;
         
  unsigned int n_well_q_points = 200,
               initial_refinement = 4;
         
  Point<2> well_center(0+excenter,0+excenter);

  //--------------------------END SETTING----------------------------------
  
  Point<2> down_left(-p_a,-p_a);
  Point<2> up_right(p_a, p_a);
  std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
  
  
  Well *well = new Well( well_radius,
                         well_center);
  well->set_perm2aquifer(0,perm2fer);
  well->set_perm2aquitard({perm2tard, 0.0});
  well->set_pressure(well_pressure);
  well->evaluate_q_points(n_well_q_points);
  
  Solution::ExactBase* exact_solution = new Solution::ExactSolution(well, radius);

    for(unsigned int refinement = 3; refinement < 8; refinement++)
    {
        std::stringstream test_name;
        test_name << "test_enr_error_" << refinement;
  XModel_simple xmodel(well);  
//   xmodel.set_name(test_name + "sgfem_model"); 
//   xmodel.set_enrichment_method(Enrichment_method::sgfem);
  xmodel.set_name(test_name.str());
  xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
  
  xmodel.set_output_dir(output_dir);
  xmodel.set_area(down_left,up_right);
  xmodel.set_transmisivity(transmisivity,0);
  xmodel.set_initial_refinement(refinement);                                     
  //xmodel.set_initial_refinement(initial_refinement);                                     
  xmodel.set_enrichment_radius(enrichment_radius);
  xmodel.set_grid_create_type(ModelBase::rect);
  xmodel.set_dirichlet_function(exact_solution);
  xmodel.set_adaptivity(true);
  //xmodel.set_well_computation_type(Well_computation::sources);
  xmodel.set_output_options(ModelBase::output_gmsh_mesh
                          //| ModelBase::output_solution
                          //| ModelBase::output_decomposed
                          | ModelBase::output_error);
  
  
  double l2_norm_dif_fem;
  std::pair<double,double> l2_norm_dif_xfem;
 
      std::cout << "===== XModel_simple running   =====" << std::endl;
      
      //xmodel.test_method(exact_solution);  
      xmodel.run();
      xmodel.test_enr_error();
      
      std::cout << "===== XModel_simple finished =====" << std::endl;
      

      Vector<double> diff_vector;
      l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, *exact_solution);
      
      xmodel.compute_interpolated_exact(exact_solution);
      xmodel.output_results();
  
        
    }
    
  delete well;
  delete exact_solution;
    
  std::cout << "\n\n:::::::::::::::: TEST ENRICHMENT RADIUS ERROR - DONE ::::::::::::::::\n\n" << std::endl;
}

int main ()
{
  std::string input_dir = "../input/";
  std::string output_dir = "../output/";
  //bedrichov_tunnel(); 
  //return 0;
  
//   test_adaptive_integration(output_dir);
  test_adaptive_integration2(output_dir);
//   test_adaptive_integration3(output_dir);
  //test_squares();
  //test_solution(output_dir);
  //test_circle_grid_creation(input_dir);
//    test_convergence_square(output_dir);
//   test_convergence_sin(output_dir);
  //test_multiple_wells(output_dir);
//   test_two_aquifers(output_dir);
  //test_output(output_dir);
//    test_enr_error(output_dir);
  return 0;
}

