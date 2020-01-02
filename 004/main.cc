
#include "xmodel.hh"

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
#include "exact_model.hh"
#include "model.hh"

#include "simple_models.hh"
#include "bem_model.hh"
#include "comparing.hh"
#include "well.hh"
#include "parameters.hh"
#include "xquadrature_cell.hh"
#include "xquadrature_well.hh"

#include "adaptive_integration.hh"
#include "global_setting_writer.hh"

#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <boost/graph/graph_concepts.hpp>
#include <boost/graph/page_rank.hpp>

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
    
    for(unsigned int i=0; i<4; i++)
    {
        std::cout << square.vertex(i)[0] << "   " << square.vertex(i)[1] << std::endl;
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


void test_convergence_square(std::string output_dir)
{
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE ::::::::::::::::\n\n" << std::endl;
    
    //------------------------------SETTING----------------------------------
    std::string test_name = "square_convergence_";
    bool //fem_create = false,
         fem = false,
         //ex = false,
         xfem = true;
    
    double p_a = 100.0,    //area of the model
           excenter = 4.89,//5.43,
           radius = p_a*std::sqrt(2),
           well_radius = 0.2,
           perm2fer = Parameters::perm2fer, 
           perm2tard = Parameters::perm2tard,
           transmisivity = Parameters::transmisivity,
           enrichment_radius = 25.6,
           well_pressure = 50*Parameters::pressure_at_top;
    
    unsigned int n_well_q_points = 1000,
                 initial_refinement = 3;
            
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
    compare::ExactWellBase* exact_solution = new compare::ExactSolution(well, radius);
    
    //FEM model creation
    Model_simple model_simple(well);  
    model_simple.set_name(test_name + "fem");
    model_simple.set_output_dir(output_dir);
    model_simple.set_area(down_left,up_right);
    model_simple.set_transmisivity(transmisivity,0);
    model_simple.set_initial_refinement(initial_refinement);  
    model_simple.set_ref_coarse_percentage(1.0,0.0);
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
//         xmodel.set_name(test_name + "xfem");
//         xmodel.set_enrichment_method(Enrichment_method::xfem);
//         xmodel.set_name(test_name + "xfem_ramp");
//         xmodel.set_enrichment_method(Enrichment_method::xfem_ramp);
//         xmodel.set_name(test_name + "xfem_shift");
//         xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
        xmodel.set_name(test_name + "sgfem"); 
        xmodel.set_enrichment_method(Enrichment_method::sgfem);
    
    xmodel.set_output_dir(output_dir);
    xmodel.set_area(down_left,up_right);
    xmodel.set_transmisivity(transmisivity,0);
    xmodel.set_initial_refinement(initial_refinement);                                     
    xmodel.set_enrichment_radius(enrichment_radius);
    xmodel.set_grid_create_type(ModelBase::rect);
    xmodel.set_dirichlet_function(exact_solution);
    xmodel.set_adaptivity(true);
    //xmodel.set_adaptive_refinement_by_error(1e-2);
    //xmodel.set_well_computation_type(Well_computation::sources);
    xmodel.set_output_options(ModelBase::output_gmsh_mesh
//                               | ModelBase::output_solution
//                               | ModelBase::output_decomposed
//                               | ModelBase::output_adaptive_plot
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
    
    
    unsigned int n_cycles = 13;
    std::cout << "Cycles: " << n_cycles << std::endl;
    std::pair<double,double> l2_norm_dif_fem, l2_norm_dif_xfem;
    
    ConvergenceTable table_convergence;
  
    for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
        table_convergence.add_value("Cycle",cycle);
        table_convergence.set_tex_format("Cycle", "r");
        double h = 2*p_a / pow(2.0,initial_refinement+cycle);
        table_convergence.add_value("h",h);
        table_convergence.set_tex_format("h", "r");
      
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
            l2_norm_dif_fem = model_simple.integrate_difference(diff_vector, exact_solution);
      
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
            l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, exact_solution);
            
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
      
            table_convergence.add_value("XFEM-dofs",
                                        xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
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


void test_radius_convergence_square(std::string output_dir)
{
  std::cout << "\n\n:::::::::::::::: RADIUS CONVERGENCE TEST ON SQUARE ::::::::::::::::\n\n" << std::endl;
  
  //------------------------------SETTING----------------------------------
  std::string test_name = "square_radius_convergence_";
  bool xfem = true;
  
  double p_a = 100.0,    //area of the model
         excenter = 5.43,
         radius = p_a*std::sqrt(2),
         well_radius = 0.2,
         perm2fer = Parameters::perm2fer, 
         perm2tard = Parameters::perm2tard,
         transmisivity = Parameters::transmisivity,
         well_pressure = 50*Parameters::pressure_at_top;
  
  unsigned int n_well_q_points = 200,
               initial_refinement = 3;
         
  Point<2> well_center(0+excenter,0+excenter);

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
  compare::ExactBase* exact_solution = new compare::ExactSolution(well, radius);
  //Function<2> *dirichlet_square = new compare::ExactSolution(well,radius);
  
    std::vector<double> enrichment_radius;
    for(unsigned int r=0; r < 4; r++)
    {
        double val = pow(2.0,r+4)*well_radius;
        enrichment_radius.push_back(val);
    }
  
    ConvergenceTable table_convergence;
    for(unsigned int r=0; r < enrichment_radius.size(); r++)
    {
        stringstream name_error, name_edofs;
        name_error << "X_L2_" << r;
        name_edofs << "edofs" << r;
        
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
        xmodel.set_enrichment_radius(enrichment_radius[r]);
        xmodel.set_grid_create_type(ModelBase::rect);
        xmodel.set_dirichlet_function(exact_solution);
        xmodel.set_adaptivity(true);
        //xmodel.set_adaptive_refinement_by_error(1e-2);
        //xmodel.set_well_computation_type(Well_computation::sources);
        xmodel.set_output_options(ModelBase::output_gmsh_mesh
        //                           | ModelBase::output_solution
        //                           | ModelBase::output_decomposed
        //                           | ModelBase::output_adaptive_plot
                                | ModelBase::output_error);
        
        
        unsigned int n_cycles = 7;
        std::cout << "Cycles: " << n_cycles << std::endl;
        std::pair<double,double> l2_norm_dif_fem, l2_norm_dif_xfem;
    
        for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
        { 
            
            if(r==0)
            {
                double h = 2*p_a / pow(2.0,initial_refinement+cycle);
                table_convergence.add_value("h",h);
                table_convergence.set_tex_format("h", "r");
            }
        
            if(xfem)
            {
                std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
        
                xmodel.run (cycle);  
        
                //xmodel.output_distributed_solution(*fine_triangulation,cycle);
                std::cout << "===== XModel_simple finished =====" << std::endl;
    
                Vector<double> diff_vector;
                l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, exact_solution);
                
                table_convergence.add_value(name_error.str(),l2_norm_dif_xfem.second);
                table_convergence.set_tex_caption(name_error.str(),"$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$");
                table_convergence.set_precision(name_error.str(), 2);
                table_convergence.set_scientific(name_error.str(),true);
                
                table_convergence.add_value(name_edofs.str(),xmodel.get_number_of_dofs().second);
                table_convergence.set_tex_format(name_edofs.str(), "r");
            }
        
            
        
    /*        if(xfem)
            {
                xmodel.compute_interpolated_exact(exact_solution);
                xmodel.output_results(cycle);
    //             ExactModel* exact = new ExactModel(exact_solution);
    //             exact->output_distributed_solution(xmodel.get_output_triangulation(), cycle);
    //             delete exact;
            }*/ 
        }   // for cycle
        
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
    }   //for r
      
  delete well;
  delete exact_solution;
  
  std::cout << "\n\n:::::::::::::::: RADIUS CONVERGENCE TEST ON SQUARE - DONE ::::::::::::::::\n\n" << std::endl;
}


void test_radius_convergence_sin(std::string output_dir)
{
    std::cout << "\n\n:::::::::::::::: RADIUS CONVERGENCE TEST ON SQUARE WITH SIN(X) ::::::::::::::::\n\n" << std::endl;
    
    //------------------------------SETTING----------------------------------
    std::string test_name = "sin_radius_convergence_";
    bool fem = true;   //compute h1 norm of the error of the regular part ur
    
    double p_a = 2.0,    //area of the model
            excenter = 0.004,
            radius = p_a*std::sqrt(2),
            well_radius = 0.003,
            perm2fer = Parameters::perm2fer, 
            perm2tard = Parameters::perm2tard,
            transmisivity = Parameters::transmisivity,
//             k_wave_num = 6,
//             amplitude = 0.02,
//             well_pressure = 9;
           k_wave_num = 1.5,
           amplitude = 0.5,
           well_pressure = 9;

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
    
    compare::ExactSolution3 *exact_solution = new compare::ExactSolution3(well, radius, k_wave_num, amplitude);
    Function<2> *dirichlet_square = exact_solution;
    Function<2> *rhs_function = new compare::Source3(*exact_solution);
  
    if (fem)    //compute h1 norm of the error of the regular part ur
    {
        //   FEM model creation
        Well *well_fem = new Well(well);
        well_fem->set_pressure(0);
        well_fem->evaluate_q_points(n_well_q_points);
        well_fem->set_inactive();
        
        compare::ExactSolution3 *exact_solution_fem = new compare::ExactSolution3(well_fem, radius, k_wave_num, amplitude);
        Function<2> *rhs_function_fem = new compare::Source3(*exact_solution_fem);
        
        Model_simple model(well_fem);  
        model.set_name(test_name + "fem");
        model.set_output_dir(output_dir);
        model.set_area(down_left,up_right);
        model.set_transmisivity(transmisivity,0);
        model.set_initial_refinement(initial_refinement-1);  
        model.set_ref_coarse_percentage(1.0,0.0);
        model.set_grid_create_type(ModelBase::rect);
        model.set_dirichlet_function(exact_solution_fem);
        model.set_rhs_function(rhs_function_fem);
        model.set_adaptivity(true);
        model.set_output_options(ModelBase::output_gmsh_mesh
                                | ModelBase::output_solution
                                | ModelBase::output_error);
        
        unsigned int n_cycles = 8;
        std::vector<double> fem_errors(n_cycles);
        for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
        { 
            model.run(cycle);
            model.output_results(cycle);
            Vector<double> diff_vector;
            std::pair<double,double> h1_norm = model.integrate_difference(diff_vector, exact_solution_fem, true);
            fem_errors[cycle] = h1_norm.second;
            std::cout << "FEM error = " << h1_norm.second << std::endl;
        }
        //write the errors down:
        std::cout << "FEM errors: " << std::endl;
        for (unsigned int i=0; i < n_cycles; ++i)
        { 
            double h = 2*p_a / pow(2.0,initial_refinement+i);
            std::cout << h << "\t" << fem_errors[i] << std::endl;
        }
        delete well_fem;
        delete rhs_function_fem;
        delete exact_solution_fem;
        return;
    }
  
    std::vector<double> enrichment_radius;
    for(unsigned int r=0; r < 10; r++)
    {
        double val = pow(2.0,r+4)*well_radius;
//         double val = 51.2;
        enrichment_radius.push_back(val);
    }
  
    ConvergenceTable table_convergence;
    for(unsigned int r=0; r < enrichment_radius.size(); r++)
    {
        stringstream name_error, name_edofs;
        name_error << "X_L2_" << r;
        name_edofs << "edofs" << r;
        
        XModel_simple xmodel(well, ""); 
        //     xmodel.set_name(test_name + "xfem");
        //     xmodel.set_enrichment_method(Enrichment_method::xfem);
        //     xmodel.set_name(test_name + "xfem_ramp");
        //     xmodel.set_enrichment_method(Enrichment_method::xfem_ramp);
//             xmodel.set_name(test_name + "xfem_shift");
//             xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
        xmodel.set_name(test_name + "sgfem"); 
        xmodel.set_enrichment_method(Enrichment_method::sgfem);
        
        xmodel.set_output_dir(output_dir);
        xmodel.set_area(down_left,up_right);
        xmodel.set_transmisivity(transmisivity,0);
        xmodel.set_initial_refinement(initial_refinement);                                     
        xmodel.set_enrichment_radius(enrichment_radius[r]);
        xmodel.set_grid_create_type(ModelBase::rect);
        xmodel.set_dirichlet_function(dirichlet_square);
        xmodel.set_rhs_function(rhs_function);
        xmodel.set_adaptivity(true);
        xmodel.set_output_options(ModelBase::output_gmsh_mesh
//                                   | ModelBase::output_solution
//                                   | ModelBase::output_decomposed
        //                           | ModelBase::output_adaptive_plot
                                | ModelBase::output_error);
        
        unsigned int n_cycles = 6;
        std::cout << "Cycles: " << n_cycles << std::endl;
        std::pair<double,double> l2_norm_dif_fem, l2_norm_dif_xfem;
    
        for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
        { 
            if(r==0)
            {
                double h = 2*p_a / pow(2.0,initial_refinement+cycle);
                table_convergence.add_value("h",h);
                table_convergence.set_tex_format("h", "r");
            }
        
            std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
        
            xmodel.run (cycle);  
        
            //xmodel.output_distributed_solution(*fine_triangulation,cycle);
            std::cout << "===== XModel_simple finished =====" << std::endl;
    
            Vector<double> diff_vector;
            l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, exact_solution);
            
            table_convergence.add_value(name_error.str(),l2_norm_dif_xfem.second);
            table_convergence.set_tex_caption(name_error.str(),"$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$");
            table_convergence.set_precision(name_error.str(), 2);
            table_convergence.set_scientific(name_error.str(),true);
            
            table_convergence.add_value(name_edofs.str(),xmodel.get_number_of_dofs().second);
            table_convergence.set_tex_format(name_edofs.str(), "r");
        }   // for cycle
        
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
    }   //for r
      
  delete well;
  delete rhs_function;
  delete exact_solution;
  
  std::cout << "\n\n:::::::::::::::: RADIUS CONVERGENCE TEST ON SQUARE WITH SIN(X) - DONE ::::::::::::::::\n\n" << std::endl;
}

void test_convergence_sin(std::string output_dir)
{
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH SIN(x) ::::::::::::::::\n\n" << std::endl;
    
    //------------------------------SETTING----------------------------------
    std::string test_name = "sin_square_convergence_";
    bool fem = false,
         xfem = true;
  
    double p_a = 100.0,    //area of the model
           excenter = 4.89,//5.43,
           radius = p_a*std::sqrt(2),
           well_radius = 0.2,
           perm2fer = Parameters::perm2fer, 
           perm2tard = Parameters::perm2tard,
           transmisivity = Parameters::transmisivity,
           k_wave_num = 0.03,
           amplitude = 8,
           well_pressure = 50*Parameters::pressure_at_top,
           enrichment_radius = 25.6;
         
    unsigned int n_well_q_points = 200,
                 initial_refinement = 3;
            
    Point<2> well_center(0+excenter,0+excenter);
    //--------------------------END SETTING----------------------------------
  
    Point<2> down_left(-p_a,-p_a);
    Point<2> up_right(p_a, p_a);
    std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
    
    Well *well = new Well( well_radius,
                            well_center);
    well->set_pressure(well_pressure);
    well->set_perm2aquifer(0,perm2fer);
    well->set_perm2aquitard({perm2tard, 0.0});
    well->evaluate_q_points(n_well_q_points);
    
    compare::ExactSolution1 *exact_solution = new compare::ExactSolution1(well, radius, k_wave_num, amplitude);
    Function<2> *dirichlet_square = exact_solution;
    Function<2> *rhs_function = new compare::Source1(*exact_solution);
    
    //FEM model creation
    Well *well_fem = new Well(well);
    //well_fem->set_pressure(0);
    //well_fem->set_inactive();
    well_fem->evaluate_q_points(n_well_q_points);
    compare::ExactSolution1 *exact_solution_fem = new compare::ExactSolution1(well_fem, radius, k_wave_num, amplitude);
    Function<2> *rhs_function_fem = new compare::Source1(*exact_solution_fem);
    Model_simple model(well_fem);  
    //Model_simple model(well);  
    model.set_name(test_name + "fem");
    model.set_output_dir(output_dir);
    model.set_area(down_left,up_right);
    model.set_transmisivity(transmisivity,0);
    model.set_initial_refinement(initial_refinement);  
    model.set_ref_coarse_percentage(1.0,0.0);
    model.set_grid_create_type(ModelBase::rect);
    //model.set_computational_mesh(coarse_file);
    model.set_dirichlet_function(exact_solution_fem);
    model.set_rhs_function(rhs_function_fem);
    model.set_adaptivity(true);
    model.set_output_options(ModelBase::output_gmsh_mesh
                            | ModelBase::output_solution
                            | ModelBase::output_error);
        
    
    XModel_simple xmodel(well);  
    xmodel.set_name(test_name + "sgfem");
    xmodel.set_enrichment_method(Enrichment_method::sgfem);
//     xmodel.set_name(test_name + "xfem_shift");
//     xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
//       xmodel.set_name(test_name + "xfem_ramp");
//       xmodel.set_enrichment_method(Enrichment_method::xfem_ramp);
//       xmodel.set_name(test_name + "xfem");
//       xmodel.set_enrichment_method(Enrichment_method::xfem);
    
    xmodel.set_output_dir(output_dir);
    xmodel.set_area(down_left,up_right);
    xmodel.set_transmisivity(transmisivity,0);
    xmodel.set_initial_refinement(initial_refinement);                                     
    xmodel.set_enrichment_radius(enrichment_radius);
    xmodel.set_grid_create_type(ModelBase::rect);
    xmodel.set_dirichlet_function(dirichlet_square);
    xmodel.set_rhs_function(rhs_function);
    xmodel.set_adaptivity(true);
    xmodel.set_output_options(ModelBase::output_gmsh_mesh
                            | ModelBase::output_solution
                            | ModelBase::output_decomposed
                            | ModelBase::output_adaptive_plot
                            | ModelBase::output_error);
    
    ExactModel exact(exact_solution);

    unsigned int n_cycles = 13;
    std::pair<double,double> l2_norm_dif_fem, l2_norm_dif_xfem;
    
    ConvergenceTable table_convergence;
  
    for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
        table_convergence.add_value("Cycle",cycle);
        table_convergence.set_tex_format("Cycle", "r");
        double h = 2*p_a / pow(2.0,initial_refinement+cycle);
        table_convergence.add_value("h",h);
        table_convergence.set_tex_format("h", "r");
        
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
            l2_norm_dif_fem = model.integrate_difference(diff_vector, exact_solution_fem);
      
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
            l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, exact_solution);
            
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
      
            table_convergence.add_value("XFEM-dofs",
                                        xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
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
//         exact.output_distributed_solution(xmodel.get_output_triangulation(), cycle);
        }
    } 
    
    delete well;
    delete exact_solution;
    delete rhs_function;
  
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH SIN(x) - DONE ::::::::::::::::\n\n" << std::endl;
}


void test_convergence_sin_2(std::string output_dir)
{
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH SIN(x) 2 ::::::::::::::::\n\n" << std::endl;
    
    //------------------------------SETTING----------------------------------
    std::string test_name = "sin_square_convergence_2_";
    bool fem = false,
         xfem = true;
  
    double p_a = 2.0,    //area of the model
           excenter = 0.004,
           radius = p_a*std::sqrt(2),
           well_radius = 0.0003,
           perm2fer = Parameters::perm2fer, 
           perm2tard = Parameters::perm2tard,
           transmisivity = Parameters::transmisivity,
           k_wave_num = 6,
           amplitude = 0.02,
           well_pressure = 9,
           enrichment_radius = 0.3;
         
    unsigned int n_well_q_points = 1000,
                 initial_refinement = 4;
            
    Point<2> well_center(0+excenter,0+excenter);
    //--------------------------END SETTING----------------------------------
  
    Point<2> down_left(-p_a,-p_a);
    Point<2> up_right(p_a, p_a);
    std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
    
    Well *well = new Well( well_radius,
                           well_center);
    well->set_pressure(well_pressure);
    well->set_perm2aquifer(0,perm2fer);
    well->set_perm2aquitard({perm2tard, 0.0});
    well->evaluate_q_points(n_well_q_points);
    
    compare::ExactSolution3 *exact_solution = new compare::ExactSolution3(well, radius, k_wave_num, amplitude);
    Function<2> *dirichlet_square = exact_solution;
    Function<2> *rhs_function = new compare::Source3(*exact_solution);
    
    //FEM model creation
    Well *well_fem = new Well(well);
    well_fem->set_pressure(0);
    well_fem->set_inactive();
    well_fem->evaluate_q_points(n_well_q_points);
    compare::ExactSolution3 *exact_solution_fem = new compare::ExactSolution3(well_fem, radius, k_wave_num, amplitude);
    Function<2> *rhs_function_fem = new compare::Source3(*exact_solution_fem);
    Model_simple model(well_fem);    
    model.set_name(test_name + "fem");
    model.set_output_dir(output_dir);
    model.set_area(down_left,up_right);
    model.set_transmisivity(transmisivity,0);
    model.set_initial_refinement(initial_refinement);  
    model.set_ref_coarse_percentage(1.0,0.0);
    model.set_grid_create_type(ModelBase::rect);
    //model.set_computational_mesh(coarse_file);
    model.set_dirichlet_function(exact_solution_fem);
    model.set_rhs_function(rhs_function_fem);
    model.set_adaptivity(true);
    model.set_output_options(ModelBase::output_gmsh_mesh
                            | ModelBase::output_solution
                            | ModelBase::output_error);
        
    
    XModel_simple xmodel(well);  
    xmodel.set_name(test_name + "sgfem");
    xmodel.set_enrichment_method(Enrichment_method::sgfem);
//     xmodel.set_name(test_name + "xfem_shift");
//     xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
//       xmodel.set_name(test_name + "xfem_ramp");
//       xmodel.set_enrichment_method(Enrichment_method::xfem_ramp);
//       xmodel.set_name(test_name + "xfem");
//       xmodel.set_enrichment_method(Enrichment_method::xfem);
    
    xmodel.set_output_dir(output_dir);
    xmodel.set_area(down_left,up_right);
    xmodel.set_transmisivity(transmisivity,0);
    xmodel.set_initial_refinement(initial_refinement);                                     
    xmodel.set_enrichment_radius(enrichment_radius);
    xmodel.set_grid_create_type(ModelBase::rect);
    xmodel.set_dirichlet_function(dirichlet_square);
    xmodel.set_rhs_function(rhs_function);
    xmodel.set_adaptivity(true);
    xmodel.set_output_options(ModelBase::output_gmsh_mesh
//                               | ModelBase::output_solution
//                               | ModelBase::output_decomposed
                            | ModelBase::output_adaptive_plot
                            | ModelBase::output_error);
    
    ExactModel exact(exact_solution);

    unsigned int n_cycles = 7;
    std::pair<double,double> l2_norm_dif_fem, l2_norm_dif_xfem;
    
    ConvergenceTable table_convergence;
  
    for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
        table_convergence.add_value("Cycle",cycle);
        table_convergence.set_tex_format("Cycle", "r");
        double h = 2*p_a / pow(2.0,initial_refinement+cycle);
        table_convergence.add_value("h",h);
        table_convergence.set_tex_format("h", "r");
        
        if(fem)
        {
            std::cout << "===== FEM Model_simple running   " << cycle << "   =====" << std::endl;
            model.run (cycle);
            model.output_results (cycle);
      
            Vector<double> diff_vector;
            l2_norm_dif_fem = model.integrate_difference(diff_vector, exact_solution_fem);
            
            compare::ExactSolutionZero zero_exact;
            std::pair<double,double> l2_norm_solution = model.integrate_difference(diff_vector, &zero_exact, true);
            std::cout << "H1 norm of solution:  " << l2_norm_solution.second << std::endl;
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
            Vector<double> diff_vector;
            l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, exact_solution);
            std::cout << "===== XModel_simple finished =====" << std::endl;
            
            table_convergence.add_value("X_L2",l2_norm_dif_xfem.second);
            table_convergence.set_tex_caption("X_L2","$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$");
            table_convergence.set_precision("X_L2", 2);
            table_convergence.set_scientific("X_L2",true);
            
            table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate);
            table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate_log2);
      
            table_convergence.add_value("XFEM-dofs",
                                        xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
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
        std::string filename = output_dir;
        if(fem) filename += model.name();
        else filename += xmodel.name();
        
        out_file.open(filename + ".tex");
        table_convergence.write_tex(out_file);
        out_file.close();
        
        out_file.open(filename + ".txt");
        table_convergence.write_text(out_file, 
                                    TableHandler::TextOutputFormat::table_with_separate_column_description);
        out_file.close();
        
        if(xfem)
        {
//         xmodel.compute_interpolated_exact(exact_solution);
        xmodel.output_results(cycle);
//         exact.output_distributed_solution(xmodel.get_output_triangulation(), cycle);
        }
    } 
    
    delete well;
    delete exact_solution;
    delete rhs_function;
    
    delete well_fem;
    delete exact_solution_fem;
    delete rhs_function_fem;
  
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH SIN(x) 2 - DONE ::::::::::::::::\n\n" << std::endl;
}


void test_convergence_sin_3(std::string output_dir)
{
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH SIN(x) 3 ::::::::::::::::\n\n" << std::endl;
    bool fem = false;
    //------------------------------SETTING----------------------------------
    std::string test_name = "sin_square_convergence_3_";
  
    double p_a = 2.0,    //area of the model
           excenter = 0.004,
           radius = p_a*std::sqrt(2),
           well_radius = 0.003,
           perm2fer = 5e1,
           perm2tard = 1e3,
           transmisivity = Parameters::transmisivity,
           k_wave_num = 1.5,
           amplitude = 0.5,
           well_pressure = 9,
           enrichment_radius = 0.3;
         
    unsigned int n_well_q_points = 300,
                 initial_refinement = 3;
            
    Point<2> well_center(0+excenter,0+excenter);
    //--------------------------END SETTING----------------------------------
  
    Point<2> down_left(-p_a,-p_a);
    Point<2> up_right(p_a, p_a);
    std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
    
    Well *well = new Well( well_radius,
                            well_center);
    well->set_pressure(well_pressure);
    well->set_perm2aquifer(0,perm2fer);
    well->set_perm2aquitard({perm2tard, 0.0});
    well->evaluate_q_points(n_well_q_points);
    

    
    if(fem) //computing H1 norm of regular part of solution
    {
        //FEM model creation
        Well *well_fem = new Well(well);
        well_fem->set_pressure(0);
        well_fem->set_inactive();
        well_fem->evaluate_q_points(n_well_q_points);
        compare::ExactSolution4 *exact_solution_fem = new compare::ExactSolution4(well_fem, radius, k_wave_num, amplitude);
        Function<2> *rhs_function_fem = new compare::Source4(*exact_solution_fem);
        Model_simple model(well_fem);    
        model.set_name(test_name + "fem");
        model.set_output_dir(output_dir);
        model.set_area(down_left,up_right);
        model.set_transmisivity(transmisivity,0);
        model.set_initial_refinement(8);  
        model.set_ref_coarse_percentage(1.0,0.0);
        model.set_grid_create_type(ModelBase::rect);
        model.set_dirichlet_function(exact_solution_fem);
        model.set_rhs_function(rhs_function_fem);
        model.set_adaptivity(true);
        model.set_output_options(ModelBase::output_gmsh_mesh
                                | ModelBase::output_solution
                                | ModelBase::output_error);
    
        model.run (0);
        model.output_results (0);
    
        Vector<double> diff_vector;
        std::pair<double,double> l2_norm_dif_fem;
        l2_norm_dif_fem = model.integrate_difference(diff_vector, exact_solution_fem);
        
        compare::ExactSolutionZero zero_exact;
        std::pair<double,double> l2_norm_solution = model.integrate_difference(diff_vector, &zero_exact, true);
        std::cout << "H1 norm of solution:  " << l2_norm_solution.second << std::endl;
        
        delete exact_solution_fem;
        delete rhs_function_fem;
        delete well_fem;
        
        delete well;
        return;
    }
        
    compare::ExactSolution4 *exact_solution = new compare::ExactSolution4(well, radius, k_wave_num, amplitude);
    Function<2> *dirichlet_square = exact_solution;
    Function<2> *rhs_function = new compare::Source4(*exact_solution);
    
    XModel_simple xmodel(well);  
//     xmodel.set_name(test_name + "sgfem");
//     xmodel.set_enrichment_method(Enrichment_method::sgfem);
    xmodel.set_name(test_name + "xfem_shift");
    xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
//       xmodel.set_name(test_name + "xfem_ramp");
//       xmodel.set_enrichment_method(Enrichment_method::xfem_ramp);
//       xmodel.set_name(test_name + "xfem");
//       xmodel.set_enrichment_method(Enrichment_method::xfem);
    
    xmodel.set_output_dir(output_dir);
    xmodel.set_area(down_left,up_right);
    xmodel.set_transmisivity(transmisivity,0);
    xmodel.set_initial_refinement(initial_refinement);                                     
    xmodel.set_enrichment_radius(enrichment_radius);
    xmodel.set_grid_create_type(ModelBase::rect);
    xmodel.set_dirichlet_function(dirichlet_square);
    xmodel.set_rhs_function(rhs_function);
    xmodel.set_adaptivity(true);
    xmodel.set_output_options(ModelBase::output_gmsh_mesh
//                               | ModelBase::output_solution
//                               | ModelBase::output_decomposed
                            | ModelBase::output_adaptive_plot
                            | ModelBase::output_error
//                             | ModelBase::output_matrix
                             );
    xmodel.set_well_band_width_ratio(0.61);
    
    ExactModel exact(exact_solution);

    unsigned int n_cycles = 7;
    std::pair<double,double> l2_norm_dif_xfem;
    
    ConvergenceTable table_convergence;
  
    for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
        table_convergence.add_value("Cycle",cycle);
        table_convergence.set_tex_format("Cycle", "r");
        double h = 2*p_a / pow(2.0,initial_refinement+cycle);
        table_convergence.add_value("h",h);
        table_convergence.set_tex_format("h", "r");
        
        std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
        xmodel.run (cycle);  
        xmodel.output_results(cycle);
        
        std::cout << "===== XModel_simple finished =====" << std::endl;
      
        Vector<double> diff_vector;
        l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, exact_solution);
        
        table_convergence.add_value("X_L2",l2_norm_dif_xfem.second);
        table_convergence.set_tex_caption("X_L2","$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$");
        table_convergence.set_precision("X_L2", 2);
        table_convergence.set_scientific("X_L2",true);
        
        table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate);
        table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate_log2);
      
        table_convergence.add_value("XFEM-dofs",
                                    xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
        table_convergence.add_value("XFEM-enriched dofs",xmodel.get_number_of_dofs().second);
        table_convergence.add_value("It_{XFEM}",xmodel.solver_iterations());
        
        table_convergence.add_value("XFEM-time",xmodel.last_run_time());
        table_convergence.set_precision("XFEM-time", 3);
        
        table_convergence.set_tex_format("XFEM-dofs", "r");
        table_convergence.set_tex_format("XFEM-enriched dofs", "r");
        table_convergence.set_tex_format("It_{XFEM}", "r");
        table_convergence.set_tex_format("XFEM-time", "r");
      
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
        
        xmodel.compute_interpolated_exact(exact_solution);
        xmodel.output_results(cycle);
//         exact.output_distributed_solution(xmodel.get_output_triangulation(), cycle);
    } 
    
        
    delete well;
    delete exact_solution;
    delete rhs_function;
  
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH SIN(x) 3 - DONE ::::::::::::::::\n\n" << std::endl;
}



void test_convergence_sin_4(std::string output_dir)
{
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH SIN(x) 4 ::::::::::::::::\n\n" << std::endl;
    bool fem = true;
    //------------------------------SETTING----------------------------------
    std::string test_name = "sin_square_convergence_4_";
  
    double p_a = 2.0,    //area of the model
           excenter = 0.004,
           radius = p_a*std::sqrt(2),
           well_radius = 0.003,
           perm2fer = 1e5,
           perm2tard = 1e10,
           
           transmisivity = 1.0,
           k_wave_num = 6,
           amplitude = 4.0,
           well_pressure = 4,
           enrichment_radius = 0.3;
           
//     double p_a = 100.0,    //area of the model
//            excenter = 5.43,
//            radius = p_a*std::sqrt(2),
// //            well_radius = 0.2,
// //            perm2fer = 1e5,
// //            perm2tard = 1e13,
//            well_radius = 0.2,
//            perm2fer = 10,
//            perm2tard = 1e10,
//            
//            transmisivity = Parameters::transmisivity,
//            k_wave_num = 0.03,
//            amplitude = 8,
//            well_pressure = 10,
//            enrichment_radius = 30;
         
    unsigned int n_well_q_points = 200,
                 initial_refinement = 3;
            
    Point<2> well_center(0+excenter,0+excenter);
    
//     std::vector<double> vec_a ({-2.57982662049939});
//     std::vector<double> vec_b ({8.16973153043276});

    //--------------------------END SETTING----------------------------------
  
    Point<2> down_left(-p_a,-p_a);
    Point<2> up_right(p_a, p_a);
    std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
    
    Well *well = new Well( well_radius,
                            well_center);
    well->set_pressure(well_pressure);
    well->set_perm2aquifer(0,perm2fer);
    well->set_perm2aquitard({perm2tard, 0.0});
    well->evaluate_q_points(n_well_q_points);
    
    std::vector<Well*> wells({well});
    
    //FEM model creation
        Well *well_fem = new Well(well);
        std::vector<Well*> wells_fem({well_fem});
        //well_fem->set_pressure(0);
//         well_fem->set_inactive();
        well_fem->evaluate_q_points(n_well_q_points);
        compare::ExactSolution4 *exact_solution_fem = new compare::ExactSolution4(well_fem, radius, k_wave_num, amplitude);
        Function<2> *rhs_function_fem = new compare::Source4(*exact_solution_fem);
        Model_simple model(well_fem);    
        model.set_name(test_name + "fem");
        model.set_output_dir(output_dir);
    if(fem) //computing H1 norm of regular part of solution
    {
        model.set_area(down_left,up_right);
        model.set_transmisivity(transmisivity,0);
        model.set_initial_refinement(4);  
//         model.set_ref_coarse_percentage(1.0,0.0);
        model.set_ref_coarse_percentage(0.4,0.0);
        model.set_grid_create_type(ModelBase::rect);
        model.set_dirichlet_function(exact_solution_fem);
        model.set_rhs_function(rhs_function_fem);
        model.set_adaptivity(true);
        model.set_output_options(ModelBase::output_gmsh_mesh
                                | ModelBase::output_solution
                                | ModelBase::output_error);
    
//         model.run (0);
//         model.output_results (0);
        
        ConvergenceTable table_convergence_fem;
        unsigned int cycles = 6;
        for(unsigned int i=0; i<cycles; i++)
        {
            table_convergence_fem.add_value("Cycle",i);
            table_convergence_fem.set_tex_format("Cycle", "r");

            model.run (i);
            model.output_results (i);
            
            Vector<double> diff_vector;
            std::pair<double,double> l2_norm_dif_fem;
            l2_norm_dif_fem = model.integrate_difference(diff_vector, exact_solution_fem);
            
            table_convergence_fem.add_value("L2",l2_norm_dif_fem.second);
            table_convergence_fem.set_tex_caption("L2","$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$");
            table_convergence_fem.set_precision("L2", 2);
            table_convergence_fem.set_scientific("L2",true);
            
            table_convergence_fem.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
            table_convergence_fem.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
            
            //write the table every cycle (to have at least some results if program fails)
            table_convergence_fem.write_text(std::cout);
            std::ofstream out_file;
            out_file.open(output_dir + model.name() + ".tex");
            table_convergence_fem.write_tex(out_file);
            out_file.close();
            
            out_file.open(output_dir + model.name() + ".txt");
            table_convergence_fem.write_text(out_file, 
                                        TableHandler::TextOutputFormat::table_with_separate_column_description);
            out_file.close();
            
            ExactModel source_output(exact_solution_fem);
            source_output.output_distributed_solution(model.get_triangulation(), i);
        }
        
        Vector<double> diff_vector;
        compare::ExactSolutionZero zero_exact;
        std::pair<double,double> l2_norm_solution = model.integrate_difference(diff_vector, &zero_exact, true);
        std::cout << "H1 norm of solution:  " << l2_norm_solution.second << std::endl;
        
//         delete exact_solution_fem;
//         delete rhs_function_fem;
//         delete well_fem;
//         
//         delete well;
//         return;
    }
        
    compare::ExactSolution4 *exact_solution = new compare::ExactSolution4(well, radius, k_wave_num, amplitude);
//     compare::ExactSolutionMultiple *exact_solution = new compare::ExactSolutionMultiple(k_wave_num, amplitude);
//     exact_solution->set_wells(wells, vec_a, vec_b);
    
    Function<2> *dirichlet_square = exact_solution;
    Function<2> *rhs_function = new compare::Source4(*exact_solution);
    
    XModel_simple xmodel(well);  
    xmodel.set_name(test_name + "sgfem");
    xmodel.set_enrichment_method(Enrichment_method::sgfem);
//     xmodel.set_name(test_name + "xfem_shift");
//     xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
//       xmodel.set_name(test_name + "xfem_ramp");
//       xmodel.set_enrichment_method(Enrichment_method::xfem_ramp);
//       xmodel.set_name(test_name + "xfem");
//       xmodel.set_enrichment_method(Enrichment_method::xfem);
    
    xmodel.set_output_dir(output_dir);
    xmodel.set_area(down_left,up_right);
    xmodel.set_transmisivity(transmisivity,0);
    xmodel.set_initial_refinement(initial_refinement);                                     
    xmodel.set_enrichment_radius(enrichment_radius);
    xmodel.set_grid_create_type(ModelBase::rect);
    xmodel.set_dirichlet_function(dirichlet_square);
    xmodel.set_rhs_function(rhs_function);
    xmodel.set_adaptivity(true);
    xmodel.set_output_options(ModelBase::output_gmsh_mesh
                              | ModelBase::output_solution
                              | ModelBase::output_decomposed
//                             | ModelBase::output_adaptive_plot
                            | ModelBase::output_error
                            | ModelBase::output_matrix
                             );
    
    ExactModel exact(exact_solution);

    unsigned int n_cycles = 7;
    std::pair<double,double> l2_norm_dif_xfem;
    
    ConvergenceTable table_convergence;
  
    for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
        table_convergence.add_value("Cycle",cycle);
        table_convergence.set_tex_format("Cycle", "r");
        double h = 2*p_a / pow(2.0,initial_refinement+cycle);
        table_convergence.add_value("h",h);
        table_convergence.set_tex_format("h", "r");
        
        std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
        xmodel.run (cycle);  
        
        std::cout << "===== XModel_simple finished =====" << std::endl;
      
        Vector<double> diff_vector;
        l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, exact_solution);
        
        table_convergence.add_value("X_L2",l2_norm_dif_xfem.second);
        table_convergence.set_tex_caption("X_L2","$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$");
        table_convergence.set_precision("X_L2", 2);
        table_convergence.set_scientific("X_L2",true);
        
        table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate);
        table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate_log2);
      
        table_convergence.add_value("XFEM-dofs",
                                    xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
        table_convergence.add_value("XFEM-enriched dofs",xmodel.get_number_of_dofs().second);
        table_convergence.add_value("It_{XFEM}",xmodel.solver_iterations());
        
        table_convergence.add_value("XFEM-time",xmodel.last_run_time());
        table_convergence.set_precision("XFEM-time", 3);
        
        table_convergence.set_tex_format("XFEM-dofs", "r");
        table_convergence.set_tex_format("XFEM-enriched dofs", "r");
        table_convergence.set_tex_format("It_{XFEM}", "r");
        table_convergence.set_tex_format("XFEM-time", "r");
      
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
        
//         xmodel.compute_interpolated_exact(exact_solution);
//         xmodel.output_results(cycle);
//         exact.output_distributed_solution(xmodel.get_output_triangulation(), cycle);
        xmodel.output_distributed_solution(model.get_triangulation(),cycle, 1);
        exact.output_distributed_solution(model.get_triangulation(), cycle);
    } 
    
        
    delete well;
    delete exact_solution;
    delete rhs_function;
  
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH SIN(x) 4 - DONE ::::::::::::::::\n\n" << std::endl;
}


void test_convergence_sin_5(std::string output_dir)
{
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH Ux^2 5 ::::::::::::::::\n\n" << std::endl;
    bool fem = false;
    //------------------------------SETTING----------------------------------
    std::string test_name = "sin_square_convergence_5_";
  
    double p_a = 2.0,    //area of the model
           excenter = 0.004,
           well_radius = 0.003,
           perm2fer = 1e5,
           perm2tard = 1e10,
           
           transmisivity = 1.0,
           amplitude = 0.7,
           well_pressure = 4.0,
           enrichment_radius = 0.3;
         
    unsigned int n_well_q_points = 200,
                 initial_refinement = 3;
            
    Point<2> well_center(0+excenter,0+excenter);
    
    double well_parameter = -0.688173755012835;

    //--------------------------END SETTING----------------------------------
  
    Point<2> down_left(-p_a,-p_a);
    Point<2> up_right(p_a, p_a);
    std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
    
    Well *well = new Well( well_radius, well_center);
    well->set_pressure(well_pressure);
    well->set_perm2aquifer(0,perm2fer);
    well->set_perm2aquitard({perm2tard, 0.0});
    well->evaluate_q_points(n_well_q_points);
    
    Well *well_fem = new Well(well);
    std::vector<Well*> wells_fem({well_fem});
    //well_fem->set_pressure(0);
//     well_fem->set_inactive();
    well_fem->evaluate_q_points(n_well_q_points);
    
    compare::ExactSolution5 *exact_solution_fem = new compare::ExactSolution5(well_fem, amplitude);
    exact_solution_fem->set_well_parameter(well_parameter);
    Function<2> *rhs_function_fem = new compare::Source5(*exact_solution_fem);
    
    //FEM model creation
    Model_simple model(well_fem);    
//     model.set_name(test_name + "fem");
    model.set_name(test_name + "fem_reg");
    model.set_output_dir(output_dir);
    if(fem) //computing H1 norm of regular part of solution
    {
        model.set_area(down_left,up_right);
        model.set_transmisivity(transmisivity,0);
        model.set_initial_refinement(initial_refinement);  
        model.set_ref_coarse_percentage(1.0,0.0);
//         model.set_ref_coarse_percentage(0.4,0.0);
        model.set_grid_create_type(ModelBase::rect);
        model.set_dirichlet_function(exact_solution_fem);
        model.set_rhs_function(rhs_function_fem);
        model.set_adaptivity(true);
        model.set_output_options(ModelBase::output_gmsh_mesh
                                | ModelBase::output_solution
                                | ModelBase::output_error);
    
//         model.run (0);
//         model.output_results (0);
        ExactModel exact_fem(exact_solution_fem);
        
        ConvergenceTable table_convergence_fem;
        unsigned int cycles = 9;
        for(unsigned int i=0; i<cycles; i++)
        {
            table_convergence_fem.add_value("Cycle",i);
            table_convergence_fem.set_tex_format("Cycle", "r");
            double h = 2*p_a / pow(2.0,initial_refinement+i);
            table_convergence_fem.add_value("h",h);
            table_convergence_fem.set_tex_format("h", "r");

            model.run (i);
            model.output_results (i);
//             exact_fem.output_distributed_solution(model.get_triangulation(), i);
            
            Vector<double> diff_vector;
            std::pair<double,double> l2_norm_dif_fem;
            l2_norm_dif_fem = model.integrate_difference(diff_vector, exact_solution_fem);
            
            table_convergence_fem.add_value("L2",l2_norm_dif_fem.second);
            table_convergence_fem.set_tex_caption("L2","$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$");
            table_convergence_fem.set_precision("L2", 2);
            table_convergence_fem.set_scientific("L2",true);
            
            table_convergence_fem.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
            table_convergence_fem.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
            
            table_convergence_fem.add_value("ndofs",model.get_number_of_dofs());
            table_convergence_fem.add_value("It_{FEM}",model.solver_iterations());
            
            //write the table every cycle (to have at least some results if program fails)
            table_convergence_fem.write_text(std::cout);
            std::ofstream out_file;
            out_file.open(output_dir + model.name() + ".tex");
            table_convergence_fem.write_tex(out_file);
            out_file.close();
            
            out_file.open(output_dir + model.name() + ".txt");
            table_convergence_fem.write_text(out_file, 
                                        TableHandler::TextOutputFormat::table_with_separate_column_description);
            out_file.close();
            
//             ExactModel source_output(exact_solution_fem);
//             source_output.output_distributed_solution(model.get_triangulation(), i);
        }
        
        Vector<double> diff_vector;
        compare::ExactSolutionZero zero_exact;
        std::pair<double,double> l2_norm_solution = model.integrate_difference(diff_vector, &zero_exact, true);
        std::cout << "H1 norm of solution:  " << l2_norm_solution.second << std::endl;
        
//         delete exact_solution_fem;
//         delete rhs_function_fem;
//         delete well_fem;
//         
//         delete well;
        return;
    }
        
    compare::ExactSolution5 *exact_solution = new compare::ExactSolution5(well, amplitude);
    exact_solution->set_well_parameter(well_parameter);
    Function<2> *rhs_function = new compare::Source5(*exact_solution);
    
    XModel_simple xmodel(well);  
    xmodel.set_name(test_name + "sgfem");
    xmodel.set_enrichment_method(Enrichment_method::sgfem);
//     xmodel.set_name(test_name + "xfem_shift");
//     xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
//       xmodel.set_name(test_name + "xfem_ramp");
//       xmodel.set_enrichment_method(Enrichment_method::xfem_ramp);
//       xmodel.set_name(test_name + "xfem");
//       xmodel.set_enrichment_method(Enrichment_method::xfem);
    
    xmodel.set_output_dir(output_dir);
    xmodel.set_area(down_left,up_right);
    xmodel.set_transmisivity(transmisivity,0);
    xmodel.set_initial_refinement(initial_refinement);                                     
    xmodel.set_enrichment_radius(enrichment_radius);
    xmodel.set_grid_create_type(ModelBase::rect);
    xmodel.set_dirichlet_function(exact_solution);
    xmodel.set_rhs_function(rhs_function);
    xmodel.set_adaptivity(true);
    xmodel.set_output_options(ModelBase::output_gmsh_mesh
                              | ModelBase::output_solution
                              | ModelBase::output_decomposed
//                             | ModelBase::output_adaptive_plot
                            | ModelBase::output_error
//                             | ModelBase::output_matrix
                             );
    
    ExactModel exact(exact_solution);

    unsigned int n_cycles = 9;
    std::pair<double,double> l2_norm_dif_xfem;
    
    ConvergenceTable table_convergence;
  
    for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
        table_convergence.add_value("Cycle",cycle);
        table_convergence.set_tex_format("Cycle", "r");
        double h = 2*p_a / pow(2.0,initial_refinement+cycle);
        table_convergence.add_value("h",h);
        table_convergence.set_tex_format("h", "r");
        
        std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
        xmodel.run (cycle);  
//         xmodel.output_results(cycle);
        
        std::cout << "===== XModel_simple finished =====" << std::endl;
      
        Vector<double> diff_vector;
        l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, exact_solution);
        
        table_convergence.add_value("X_L2",l2_norm_dif_xfem.second);
        table_convergence.set_tex_caption("X_L2","$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$");
        table_convergence.set_precision("X_L2", 2);
        table_convergence.set_scientific("X_L2",true);
        
        table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate);
        table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate_log2);
      
        table_convergence.add_value("XFEM-dofs",
                                    xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
        table_convergence.add_value("XFEM-enriched dofs",xmodel.get_number_of_dofs().second);
        table_convergence.add_value("It_{XFEM}",xmodel.solver_iterations());
        
        table_convergence.add_value("XFEM-time",xmodel.last_run_time());
        table_convergence.set_precision("XFEM-time", 3);
        
        table_convergence.set_tex_format("XFEM-dofs", "r");
        table_convergence.set_tex_format("XFEM-enriched dofs", "r");
        table_convergence.set_tex_format("It_{XFEM}", "r");
        table_convergence.set_tex_format("XFEM-time", "r");
      
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
        
//         xmodel.compute_interpolated_exact(exact_solution);
//         xmodel.output_results(cycle);
//         exact.output_distributed_solution(xmodel.get_output_triangulation(), cycle);
//         xmodel.output_distributed_solution(model.get_triangulation(),cycle, 1);
//         exact.output_distributed_solution(model.get_triangulation(), cycle);
    } 
    
        
    delete well;
    delete exact_solution;
    delete rhs_function;
  
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH Ux^2 5 - DONE ::::::::::::::::\n\n" << std::endl;
}

void test_square_rhow(bool fem, unsigned int cycle, double excenter, double well_parameter,
                      ConvergenceTable &table_convergence, std::string output_dir)
{
    //------------------------------SETTING----------------------------------
    std::string test_name = "square_5_rhow";
  
    double p_a = 2.0,    //area of the model
           well_radius = 0.003,
           perm2fer = 1e5,
           perm2tard = 1e10,
           
           transmisivity = 1.0,
           amplitude = 0.7,
           well_pressure = 4.0,
           enrichment_radius = 0.5;
         
    unsigned int n_well_q_points = 200,
                 initial_refinement = 5;

    //--------------------------END SETTING----------------------------------
  
    Point<2> down_left(-p_a,-p_a);
    Point<2> up_right(p_a, p_a);
    std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
    
        Point<2> well_center(0+excenter,0+excenter);
        Well *well = new Well( well_radius, well_center);
        well->set_pressure(well_pressure);
        well->set_perm2aquifer(0,perm2fer);
        well->set_perm2aquitard({perm2tard, 0.0});
        well->evaluate_q_points(n_well_q_points);

        compare::ExactSolution5 *exact_solution = new compare::ExactSolution5(well, amplitude);
        exact_solution->set_well_parameter(well_parameter);
        Function<2> *rhs_function = new compare::Source5(*exact_solution);


        
        
    //FEM model creation
    if(fem) //computing H1 norm of regular part of solution
    {
        Well *well_fem = new Well(well);
        std::vector<Well*> wells_fem({well_fem});
        well_fem->set_inactive();
        well_fem->evaluate_q_points(n_well_q_points);

        compare::ExactSolution5 *exact_solution_fem = new compare::ExactSolution5(well_fem, amplitude);
        exact_solution_fem->set_well_parameter(well_parameter);
        Function<2> *rhs_function_fem = new compare::Source5(*exact_solution_fem);
    
        Model_simple model(well_fem);    
//     model.set_name(test_name + "fem");
        model.set_name(test_name + "fem_reg");
        model.set_output_dir(output_dir);
        model.set_area(down_left,up_right);
        model.set_transmisivity(transmisivity,0);
        model.set_initial_refinement(initial_refinement);  
        model.set_ref_coarse_percentage(1.0,0.0);
//         model.set_ref_coarse_percentage(0.4,0.0);
        model.set_grid_create_type(ModelBase::rect);
        model.set_dirichlet_function(exact_solution_fem);
        model.set_rhs_function(rhs_function_fem);
        model.set_adaptivity(false);
        model.set_output_options(ModelBase::output_gmsh_mesh
                                | ModelBase::output_solution
                                | ModelBase::output_error);
    
//         model.run (0);
//         model.output_results (0);
        ExactModel exact_fem(exact_solution_fem);
        
            table_convergence.add_value("Cycle",cycle);
            table_convergence.set_tex_format("Cycle", "r");
            double h = 2*p_a / pow(2.0,initial_refinement);
            table_convergence.add_value("h",h);
            table_convergence.set_tex_format("h", "r");

            model.run (0);
//             model.output_results (0);
//             exact_fem.output_distributed_solution(model.get_triangulation(), i);
            
            Vector<double> diff_vector;
            std::pair<double,double> l2_norm_dif_fem;
            l2_norm_dif_fem = model.integrate_difference(diff_vector, exact_solution_fem);
            
            table_convergence.add_value("L2",l2_norm_dif_fem.second);
            table_convergence.set_tex_caption("L2","$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$");
            table_convergence.set_precision("L2", 2);
            table_convergence.set_scientific("L2",true);
            
            table_convergence.add_value("ndofs",model.get_number_of_dofs());
            table_convergence.add_value("It_{FEM}",model.solver_iterations());
            
            //write the table every cycle (to have at least some results if program fails)
            table_convergence.write_text(std::cout);
            std::ofstream out_file;
            out_file.open(output_dir + model.name() + ".tex");
            table_convergence.write_tex(out_file);
            out_file.close();
            
            out_file.open(output_dir + model.name() + ".txt");
            table_convergence.write_text(out_file, 
                                        TableHandler::TextOutputFormat::table_with_separate_column_description);
            out_file.close();
            
//         compare::ExactSolutionZero zero_exact;
//         std::pair<double,double> l2_norm_solution = model.integrate_difference(diff_vector, &zero_exact, true);
//         std::cout << "H1 norm of solution:  " << l2_norm_solution.second << std::endl;
        
//         delete exact_solution_fem;
//         delete rhs_function_fem;
//         delete well_fem;
//         
//         delete well;
        return;
    }
        
        
        
        
        
        XModel_simple xmodel(well);
        xmodel.set_name(test_name + "sgfem");
        xmodel.set_enrichment_method(Enrichment_method::sgfem);
    //     xmodel.set_name(test_name + "xfem_shift");
    //     xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
    //       xmodel.set_name(test_name + "xfem_ramp");
    //       xmodel.set_enrichment_method(Enrichment_method::xfem_ramp);
    //       xmodel.set_name(test_name + "xfem");
    //       xmodel.set_enrichment_method(Enrichment_method::xfem);
    
        xmodel.set_output_dir(output_dir);
        xmodel.set_area(down_left,up_right);
        xmodel.set_transmisivity(transmisivity,0);
        xmodel.set_initial_refinement(initial_refinement);                                     
        xmodel.set_enrichment_radius(enrichment_radius);
        xmodel.set_grid_create_type(ModelBase::rect);
        xmodel.set_dirichlet_function(exact_solution);
        xmodel.set_rhs_function(rhs_function);
        xmodel.set_adaptivity(false);
        xmodel.set_output_options(ModelBase::output_gmsh_mesh
                                | ModelBase::output_solution
                                | ModelBase::output_decomposed
    //                             | ModelBase::output_adaptive_plot
                                | ModelBase::output_error
    //                             | ModelBase::output_matrix
                                );
    
        ExactModel exact(exact_solution);
    
        table_convergence.add_value("Cycle",cycle);
        table_convergence.set_tex_format("Cycle", "r");
        double h = 2*p_a / pow(2.0,initial_refinement);
        table_convergence.add_value("h",h);
        table_convergence.set_tex_format("h", "r");
        
        std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
        xmodel.run (0);  
//         xmodel.output_results(cycle);
        
        std::cout << "===== XModel_simple finished =====" << std::endl;
      
        std::pair<double,double> l2_norm_dif_xfem;
        Vector<double> diff_vector;
        l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, exact_solution);
        
        std::cout << "ERROR" << std::endl;
        
        table_convergence.add_value("X_L2",l2_norm_dif_xfem.second);
        table_convergence.set_tex_caption("X_L2","$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$");
        table_convergence.set_precision("X_L2", 2);
        table_convergence.set_scientific("X_L2",true);
      
        table_convergence.add_value("XFEM-dofs",
                                    xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
        table_convergence.add_value("XFEM-enriched dofs",xmodel.get_number_of_dofs().second);
        table_convergence.add_value("It_{XFEM}",xmodel.solver_iterations());
        
        table_convergence.set_tex_format("XFEM-dofs", "r");
        table_convergence.set_tex_format("XFEM-enriched dofs", "r");
        table_convergence.set_tex_format("It_{XFEM}", "r");
        table_convergence.set_tex_format("XFEM-time", "r");
        std::cout << "ERROR" << std::endl;
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
        std::cout << "ERROR" << std::endl;
//         xmodel.compute_interpolated_exact(exact_solution);
//         xmodel.output_results(cycle);
//         exact.output_distributed_solution(xmodel.get_output_triangulation(), cycle);
//         xmodel.output_distributed_solution(model.get_triangulation(),cycle, 1);
//         exact.output_distributed_solution(model.get_triangulation(), cycle);

        delete rhs_function;
        delete exact_solution;
        delete well;
        std::cout << "ERROR" << std::endl;
}

void test_convergence_5_rhow(std::string output_dir)
{
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH Ux^2 5 RHO_W ::::::::::::::::\n\n" << std::endl;
    
    std::vector<double> excenter ({0.0, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.015, 0.03, 0.06, 0.1, 0.11, 0.120, 0.123, 0.125});
    double well_parameter = -0.688168611249941;
//     std::vector<double> well_parameter ({-0.688174846114054,  -0.679917619489053,  -0.675789377751973,  -0.671661581890416, -0.663407922173293,  -0.655157828850421,
//   -0.646912489920981,  -0.626327732672823,  -0.564981242792659,  -0.445768346874862, -0.299633782451967});

    //--------------------------END SETTING----------------------------------

    
    ConvergenceTable table_convergence;
    
    unsigned int n_cycles = excenter.size();
    for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
        test_square_rhow(true, cycle, excenter[cycle], well_parameter, table_convergence, output_dir);
    } 
    
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH Ux^2 5 RHO_W - DONE ::::::::::::::::\n\n" << std::endl;
}


// This is not converging because the suggested source term include a soft singularity
// so the standard FEM does not converge either.
void test_convergence_sin_6(std::string output_dir)
{
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH SIN(x) 6 ::::::::::::::::\n\n" << std::endl;
    bool fem = true;
    //------------------------------SETTING----------------------------------
    std::string test_name = "sin_square_convergence_6_";
  
    double p_a = 2.0,    //area of the model
           excenter = 0.004,
           well_radius = 0.09,
           perm2fer = 1e5,
           perm2tard = 1e10,
           
           transmisivity = 1.0,
           k_wave_num = 6,
           amplitude = 0.7,
           well_pressure = 1,//9,
           enrichment_radius = 0.3;
         
    unsigned int n_well_q_points = 200,
                 initial_refinement = 3;
            
    Point<2> well_center(0+excenter,0+excenter);
    
    double well_parameter = -0.415269703363931;

    //--------------------------END SETTING----------------------------------
  
    Point<2> down_left(-p_a,-p_a);
    Point<2> up_right(p_a, p_a);
    std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
    
    Well *well = new Well( well_radius, well_center);
    well->set_pressure(well_pressure);
    well->set_perm2aquifer(0,perm2fer);
    well->set_perm2aquitard({perm2tard, 0.0});
    well->evaluate_q_points(n_well_q_points);
    
    Well *well_fem = new Well(well);
    std::vector<Well*> wells_fem({well_fem});
    //well_fem->set_pressure(0);
    well_fem->set_inactive();
    well_fem->evaluate_q_points(n_well_q_points);
    
    compare::ExactSolution6 *exact_solution_fem = new compare::ExactSolution6(well_fem, k_wave_num, amplitude);
    exact_solution_fem->set_well_parameter(well_parameter);
    Function<2> *rhs_function_fem = new compare::Source6(*exact_solution_fem);
    
    //FEM model creation
    Model_simple model(well_fem);    
    model.set_name(test_name + "fem");
    model.set_output_dir(output_dir);
    if(fem) //computing H1 norm of regular part of solution
    {
        model.set_area(down_left,up_right);
        model.set_transmisivity(transmisivity,0);
        model.set_initial_refinement(4);  
        model.set_ref_coarse_percentage(1.0,0.0);
//         model.set_ref_coarse_percentage(0.4,0.0);
        model.set_grid_create_type(ModelBase::rect);
        model.set_dirichlet_function(exact_solution_fem);
        model.set_rhs_function(rhs_function_fem);
        model.set_adaptivity(true);
        model.set_output_options(ModelBase::output_gmsh_mesh
                                | ModelBase::output_solution
                                | ModelBase::output_error);
    
//         model.run (0);
//         model.output_results (0);
        ExactModel exact_fem(exact_solution_fem);
        
        ConvergenceTable table_convergence_fem;
        unsigned int cycles = 6;
        for(unsigned int i=0; i<cycles; i++)
        {
            table_convergence_fem.add_value("Cycle",i);
            table_convergence_fem.set_tex_format("Cycle", "r");

            model.run (i);
            model.output_results (i);
            exact_fem.output_distributed_solution(model.get_triangulation(), i);
            
            Vector<double> diff_vector;
            std::pair<double,double> l2_norm_dif_fem;
            l2_norm_dif_fem = model.integrate_difference(diff_vector, exact_solution_fem);
            
            table_convergence_fem.add_value("L2",l2_norm_dif_fem.second);
            table_convergence_fem.set_tex_caption("L2","$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$");
            table_convergence_fem.set_precision("L2", 2);
            table_convergence_fem.set_scientific("L2",true);
            
            table_convergence_fem.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
            table_convergence_fem.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
            
            //write the table every cycle (to have at least some results if program fails)
            table_convergence_fem.write_text(std::cout);
            std::ofstream out_file;
            out_file.open(output_dir + model.name() + ".tex");
            table_convergence_fem.write_tex(out_file);
            out_file.close();
            
            out_file.open(output_dir + model.name() + ".txt");
            table_convergence_fem.write_text(out_file, 
                                        TableHandler::TextOutputFormat::table_with_separate_column_description);
            out_file.close();
        }
        
        Vector<double> diff_vector;
        compare::ExactSolutionZero zero_exact;
        std::pair<double,double> l2_norm_solution = model.integrate_difference(diff_vector, &zero_exact, true);
        std::cout << "H1 norm of solution:  " << l2_norm_solution.second << std::endl;
        
//         delete exact_solution_fem;
//         delete rhs_function_fem;
//         delete well_fem;
//         
//         delete well;
        return;
    }
        
    compare::ExactSolution6 *exact_solution = new compare::ExactSolution6(well, k_wave_num, amplitude);
    exact_solution->set_well_parameter(well_parameter);
    Function<2> *rhs_function = new compare::Source6(*exact_solution);
    
    XModel_simple xmodel(well);  
    xmodel.set_name(test_name + "sgfem");
    xmodel.set_enrichment_method(Enrichment_method::sgfem);
//     xmodel.set_name(test_name + "xfem_shift");
//     xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
//       xmodel.set_name(test_name + "xfem_ramp");
//       xmodel.set_enrichment_method(Enrichment_method::xfem_ramp);
//       xmodel.set_name(test_name + "xfem");
//       xmodel.set_enrichment_method(Enrichment_method::xfem);
    
    xmodel.set_output_dir(output_dir);
    xmodel.set_area(down_left,up_right);
    xmodel.set_transmisivity(transmisivity,0);
    xmodel.set_initial_refinement(initial_refinement);                                     
    xmodel.set_enrichment_radius(enrichment_radius);
    xmodel.set_grid_create_type(ModelBase::rect);
    xmodel.set_dirichlet_function(exact_solution);
    xmodel.set_rhs_function(rhs_function);
    xmodel.set_adaptivity(true);
    xmodel.set_output_options(ModelBase::output_gmsh_mesh
                              | ModelBase::output_solution
                              | ModelBase::output_decomposed
//                             | ModelBase::output_adaptive_plot
                            | ModelBase::output_error
                            | ModelBase::output_matrix
                             );
    
    ExactModel exact(exact_solution);

    unsigned int n_cycles = 7;
    std::pair<double,double> l2_norm_dif_xfem;
    
    ConvergenceTable table_convergence;
  
    for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
        table_convergence.add_value("Cycle",cycle);
        table_convergence.set_tex_format("Cycle", "r");
        double h = 2*p_a / pow(2.0,initial_refinement+cycle);
        table_convergence.add_value("h",h);
        table_convergence.set_tex_format("h", "r");
        
        std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
        xmodel.run (cycle);  
//         xmodel.output_results(cycle);
        
        std::cout << "===== XModel_simple finished =====" << std::endl;
      
        Vector<double> diff_vector;
        l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, exact_solution);
        
        table_convergence.add_value("X_L2",l2_norm_dif_xfem.second);
        table_convergence.set_tex_caption("X_L2","$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$");
        table_convergence.set_precision("X_L2", 2);
        table_convergence.set_scientific("X_L2",true);
        
        table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate);
        table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate_log2);
      
        table_convergence.add_value("XFEM-dofs",
                                    xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
        table_convergence.add_value("XFEM-enriched dofs",xmodel.get_number_of_dofs().second);
        table_convergence.add_value("It_{XFEM}",xmodel.solver_iterations());
        
        table_convergence.add_value("XFEM-time",xmodel.last_run_time());
        table_convergence.set_precision("XFEM-time", 3);
        
        table_convergence.set_tex_format("XFEM-dofs", "r");
        table_convergence.set_tex_format("XFEM-enriched dofs", "r");
        table_convergence.set_tex_format("It_{XFEM}", "r");
        table_convergence.set_tex_format("XFEM-time", "r");
      
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
        
//         xmodel.compute_interpolated_exact(exact_solution);
//         xmodel.output_results(cycle);
//         exact.output_distributed_solution(xmodel.get_output_triangulation(), cycle);
        xmodel.output_distributed_solution(model.get_triangulation(),cycle, 1);
        exact.output_distributed_solution(model.get_triangulation(), cycle);
    } 
    
        
    delete well;
    delete exact_solution;
    delete rhs_function;
  
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH SIN(x) 6 - DONE ::::::::::::::::\n\n" << std::endl;
}

void test_convergence_sin_7(std::string output_dir)
{
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH SIN(x) 7 ::::::::::::::::\n\n" << std::endl;
    bool fem = true;
    //------------------------------SETTING----------------------------------
    std::string test_name = "sin_square_convergence_7_X_";
  
    double p_a = 2.0,    //area of the model
           excenter = 0.004,
           well_radius = 0.003,
           perm2fer = 1e5,
           perm2tard = 1e10,
           
           transmisivity = 1.0,
           k_wave_num = 6,
           amplitude = 4,
           well_pressure = 4.0,
           enrichment_radius = 0.3;
         
    unsigned int n_well_q_points = 200,
                 initial_refinement = 3;
            
    Point<2> well_center(0+excenter,0+excenter);
    
    double well_parameter = -0.671661581890416;

    //--------------------------END SETTING----------------------------------
  
    Point<2> down_left(-p_a,-p_a);
    Point<2> up_right(p_a, p_a);
    std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
    
    Well *well = new Well( well_radius, well_center);
    well->set_pressure(well_pressure);
    well->set_perm2aquifer(0,perm2fer);
    well->set_perm2aquitard({perm2tard, 0.0});
    well->evaluate_q_points(n_well_q_points);
    
    Well *well_fem = new Well(well);
    std::vector<Well*> wells_fem({well_fem});
    //well_fem->set_pressure(0);
    well_fem->set_inactive();
    well_fem->evaluate_q_points(n_well_q_points);
    
    compare::ExactSolution7 *exact_solution_fem = new compare::ExactSolution7(well_fem, k_wave_num, amplitude);
    exact_solution_fem->set_well_parameter(well_parameter);
    Function<2> *rhs_function_fem = new compare::Source7(*exact_solution_fem);
    
    //FEM model creation
    Model_simple model(well_fem);    
//     model.set_name(test_name + "fem");
    model.set_name(test_name + "fem_reg");
    model.set_output_dir(output_dir);
    if(fem) //computing H1 norm of regular part of solution
    {
        model.set_area(down_left,up_right);
        model.set_transmisivity(transmisivity,0);
        model.set_initial_refinement(initial_refinement);  
        model.set_ref_coarse_percentage(1.0,0.0);
//         model.set_ref_coarse_percentage(0.4,0.0);
        model.set_grid_create_type(ModelBase::rect);
        model.set_dirichlet_function(exact_solution_fem);
        model.set_rhs_function(rhs_function_fem);
        model.set_adaptivity(true);
        model.set_output_options(ModelBase::output_gmsh_mesh
                                | ModelBase::output_solution
                                | ModelBase::output_error);
    
//         model.run (0);
//         model.output_results (0);
        ExactModel exact_fem(exact_solution_fem);
        
        ConvergenceTable table_convergence_fem;
        unsigned int cycles = 8;
        for(unsigned int i=0; i<cycles; i++)
        {
            table_convergence_fem.add_value("Cycle",i);
            table_convergence_fem.set_tex_format("Cycle", "r");

            model.run (i);
//             model.output_results (i);
//             exact_fem.output_distributed_solution(model.get_triangulation(), i);
            
            Vector<double> diff_vector;
            std::pair<double,double> l2_norm_dif_fem;
            l2_norm_dif_fem = model.integrate_difference(diff_vector, exact_solution_fem);
            
            table_convergence_fem.add_value("L2",l2_norm_dif_fem.second);
            table_convergence_fem.set_tex_caption("L2","$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$");
            table_convergence_fem.set_precision("L2", 2);
            table_convergence_fem.set_scientific("L2",true);
            
            table_convergence_fem.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
            table_convergence_fem.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
            
            //write the table every cycle (to have at least some results if program fails)
            table_convergence_fem.write_text(std::cout);
            std::ofstream out_file;
            out_file.open(output_dir + model.name() + ".tex");
            table_convergence_fem.write_tex(out_file);
            out_file.close();
            
            out_file.open(output_dir + model.name() + ".txt");
            table_convergence_fem.write_text(out_file, 
                                        TableHandler::TextOutputFormat::table_with_separate_column_description);
            out_file.close();
            
//             ExactModel source_output(exact_solution_fem);
//             source_output.output_distributed_solution(model.get_triangulation(), i);
        }
        
        Vector<double> diff_vector;
        compare::ExactSolutionZero zero_exact;
        std::pair<double,double> l2_norm_solution = model.integrate_difference(diff_vector, &zero_exact, true);
        std::cout << "H1 norm of solution:  " << l2_norm_solution.second << std::endl;
        
//         delete exact_solution_fem;
//         delete rhs_function_fem;
//         delete well_fem;
//         
//         delete well;
        return;
    }
        
    compare::ExactSolution7 *exact_solution = new compare::ExactSolution7(well, k_wave_num, amplitude);
    exact_solution->set_well_parameter(well_parameter);
    Function<2> *rhs_function = new compare::Source7(*exact_solution);
    
    XModel_simple xmodel(well);  
//     xmodel.set_name(test_name + "sgfem");
//     xmodel.set_enrichment_method(Enrichment_method::sgfem);
    xmodel.set_name(test_name + "xfem_shift");
    xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
//       xmodel.set_name(test_name + "xfem_ramp");
//       xmodel.set_enrichment_method(Enrichment_method::xfem_ramp);
//       xmodel.set_name(test_name + "xfem");
//       xmodel.set_enrichment_method(Enrichment_method::xfem);
    
    xmodel.set_output_dir(output_dir);
    xmodel.set_area(down_left,up_right);
    xmodel.set_transmisivity(transmisivity,0);
    xmodel.set_initial_refinement(initial_refinement);                                     
    xmodel.set_enrichment_radius(enrichment_radius);
    xmodel.set_grid_create_type(ModelBase::rect);
    xmodel.set_dirichlet_function(exact_solution);
    xmodel.set_rhs_function(rhs_function);
    xmodel.set_adaptivity(true);
    xmodel.set_output_options(ModelBase::output_gmsh_mesh
                              | ModelBase::output_solution
                              | ModelBase::output_decomposed
//                             | ModelBase::output_adaptive_plot
                            | ModelBase::output_error
//                             | ModelBase::output_matrix
                             );
    
    ExactModel exact(exact_solution);

    unsigned int n_cycles = 8;
    std::pair<double,double> l2_norm_dif_xfem;
    
    ConvergenceTable table_convergence;
  
    for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
        table_convergence.add_value("Cycle",cycle);
        table_convergence.set_tex_format("Cycle", "r");
        double h = 2*p_a / pow(2.0,initial_refinement+cycle);
        table_convergence.add_value("h",h);
        table_convergence.set_tex_format("h", "r");
        
        std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
        xmodel.run (cycle);  
//         xmodel.output_results(cycle);
        
        std::cout << "===== XModel_simple finished =====" << std::endl;
      
        Vector<double> diff_vector;
        l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, exact_solution);
        
        table_convergence.add_value("X_L2",l2_norm_dif_xfem.second);
        table_convergence.set_tex_caption("X_L2","$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$");
        table_convergence.set_precision("X_L2", 2);
        table_convergence.set_scientific("X_L2",true);
        
        table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate);
        table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate_log2);
      
        table_convergence.add_value("XFEM-dofs",
                                    xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
        table_convergence.add_value("XFEM-enriched dofs",xmodel.get_number_of_dofs().second);
        table_convergence.add_value("It_{XFEM}",xmodel.solver_iterations());
        
        table_convergence.add_value("XFEM-time",xmodel.last_run_time());
        table_convergence.set_precision("XFEM-time", 3);
        
        table_convergence.set_tex_format("XFEM-dofs", "r");
        table_convergence.set_tex_format("XFEM-enriched dofs", "r");
        table_convergence.set_tex_format("It_{XFEM}", "r");
        table_convergence.set_tex_format("XFEM-time", "r");
      
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
        
//         xmodel.compute_interpolated_exact(exact_solution);
//         xmodel.output_results(cycle);
//         exact.output_distributed_solution(xmodel.get_output_triangulation(), cycle);
//         xmodel.output_distributed_solution(model.get_triangulation(),cycle, 1);
//         exact.output_distributed_solution(model.get_triangulation(), cycle);
    } 
    
        
    delete well;
    delete exact_solution;
    delete rhs_function;
  
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH SIN(x) 7 - DONE ::::::::::::::::\n\n" << std::endl;
}

void test_square_sin_7_rhow_func(bool fem, unsigned int cycle, double excenter, double well_parameter,
                      ConvergenceTable &table_convergence, std::string output_dir)
{
    //------------------------------SETTING----------------------------------
    std::string test_name = "square_sin_7_rhow_";
  
    double p_a = 2.0,    //area of the model
           well_radius = 0.003,
           perm2fer = 1e5,
           perm2tard = 1e10,
           
           transmisivity = 1.0,
           k_wave_num = 6,
           
//            amplitude = 4,
           amplitude = 0,
           
           well_pressure = 4.0,
           enrichment_radius = 0.3;
         
    well_parameter = -0.688174846114054;
    
    unsigned int n_well_q_points = 200,
                 initial_refinement = 5;

    //--------------------------END SETTING----------------------------------
  
    Point<2> down_left(-p_a,-p_a);
    Point<2> up_right(p_a, p_a);
    std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
    
    Point<2> well_center(0+excenter,0+excenter);
    Well *well = new Well( well_radius, well_center);
    well->set_pressure(well_pressure);
    well->set_perm2aquifer(0,perm2fer);
    well->set_perm2aquitard({perm2tard, 0.0});
    well->evaluate_q_points(n_well_q_points);
        
    //FEM model creation
    if(fem) //computing H1 norm of regular part of solution
    {
        Well *well_fem = new Well(well);
        std::vector<Well*> wells_fem({well_fem});
        well_fem->set_inactive();
        well_fem->evaluate_q_points(n_well_q_points);

        compare::ExactSolution7 *exact_solution_fem = new compare::ExactSolution7(well_fem, k_wave_num, amplitude);
        exact_solution_fem->set_well_parameter(well_parameter);
        Function<2> *rhs_function_fem = new compare::Source7(*exact_solution_fem);
    
        Model_simple model(well_fem);    
//     model.set_name(test_name + "fem");
        model.set_name(test_name + "fem_reg");
        model.set_output_dir(output_dir);
        model.set_area(down_left,up_right);
        model.set_transmisivity(transmisivity,0);
        model.set_initial_refinement(initial_refinement);  
        model.set_ref_coarse_percentage(1.0,0.0);
//         model.set_ref_coarse_percentage(0.4,0.0);
        model.set_grid_create_type(ModelBase::rect);
        model.set_dirichlet_function(exact_solution_fem);
        model.set_rhs_function(rhs_function_fem);
        model.set_adaptivity(false);
        model.set_output_options(ModelBase::output_gmsh_mesh
                                | ModelBase::output_solution
                                | ModelBase::output_error);
    
//         model.run (0);
//         model.output_results (0);
        ExactModel exact_fem(exact_solution_fem);
        
            table_convergence.add_value("Cycle",cycle);
            table_convergence.set_tex_format("Cycle", "r");
            double h = 2*p_a / pow(2.0,initial_refinement);
            table_convergence.add_value("h",h);
            table_convergence.set_tex_format("h", "r");

            model.run (0);
//             model.output_results (0);
//             exact_fem.output_distributed_solution(model.get_triangulation(), i);
            
            Vector<double> diff_vector;
            std::pair<double,double> l2_norm_dif_fem;
            l2_norm_dif_fem = model.integrate_difference(diff_vector, exact_solution_fem);
            
            table_convergence.add_value("L2",l2_norm_dif_fem.second);
            table_convergence.set_tex_caption("L2","$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$");
            table_convergence.set_precision("L2", 2);
            table_convergence.set_scientific("L2",true);
            
            table_convergence.add_value("ndofs",model.get_number_of_dofs());
            table_convergence.add_value("It_{FEM}",model.solver_iterations());
            
            //write the table every cycle (to have at least some results if program fails)
            table_convergence.write_text(std::cout);
            std::ofstream out_file;
            out_file.open(output_dir + model.name() + ".tex");
            table_convergence.write_tex(out_file);
            out_file.close();
            
            out_file.open(output_dir + model.name() + ".txt");
            table_convergence.write_text(out_file, 
                                        TableHandler::TextOutputFormat::table_with_separate_column_description);
            out_file.close();
            
//         compare::ExactSolutionZero zero_exact;
//         std::pair<double,double> l2_norm_solution = model.integrate_difference(diff_vector, &zero_exact, true);
//         std::cout << "H1 norm of solution:  " << l2_norm_solution.second << std::endl;
        
//         delete exact_solution_fem;
//         delete rhs_function_fem;
//         delete well_fem;
//         
//         delete well;
        return;
    }
        
        
        
    compare::ExactSolution7 *exact_solution = new compare::ExactSolution7(well, k_wave_num, amplitude);
    exact_solution->set_well_parameter(well_parameter);
    Function<2> *rhs_function = new compare::Source7(*exact_solution);
        
    XModel_simple xmodel(well);
//     xmodel.set_name(test_name + "sgfem");
//     xmodel.set_enrichment_method(Enrichment_method::sgfem);
//         xmodel.set_name(test_name + "xfem_shift");
//         xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
//           xmodel.set_name(test_name + "xfem_ramp");
//           xmodel.set_enrichment_method(Enrichment_method::xfem_ramp);
          xmodel.set_name(test_name + "xfem");
          xmodel.set_enrichment_method(Enrichment_method::xfem);
    
        xmodel.set_output_dir(output_dir);
        xmodel.set_area(down_left,up_right);
        xmodel.set_transmisivity(transmisivity,0);
        xmodel.set_initial_refinement(initial_refinement);                                     
        xmodel.set_enrichment_radius(enrichment_radius);
        xmodel.set_grid_create_type(ModelBase::rect);
        xmodel.set_dirichlet_function(exact_solution);
        xmodel.set_rhs_function(rhs_function);
        xmodel.set_adaptivity(false);
        xmodel.set_output_options(ModelBase::output_gmsh_mesh
                                | ModelBase::output_solution
                                | ModelBase::output_decomposed
    //                             | ModelBase::output_adaptive_plot
                                | ModelBase::output_error
    //                             | ModelBase::output_matrix
                                );
    
        ExactModel exact(exact_solution);
    
        table_convergence.add_value("Cycle",cycle);
        table_convergence.set_tex_format("Cycle", "r");
        double h = 2*p_a / pow(2.0,initial_refinement);
        table_convergence.add_value("h",h);
        table_convergence.set_tex_format("h", "r");
        
        std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
        xmodel.run (0);  
//         xmodel.output_results(cycle);
        
        std::cout << "===== XModel_simple finished =====" << std::endl;
      
        std::pair<double,double> l2_norm_dif_xfem;
        Vector<double> diff_vector;
        l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, exact_solution);
        
        
        table_convergence.add_value("X_L2",l2_norm_dif_xfem.second);
        table_convergence.set_tex_caption("X_L2","$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$");
        table_convergence.set_precision("X_L2", 2);
        table_convergence.set_scientific("X_L2",true);
      
        table_convergence.add_value("XFEM-dofs",
                                    xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
        table_convergence.add_value("XFEM-enriched dofs",xmodel.get_number_of_dofs().second);
        table_convergence.add_value("It_{XFEM}",xmodel.solver_iterations());
        
        table_convergence.set_tex_format("XFEM-dofs", "r");
        table_convergence.set_tex_format("XFEM-enriched dofs", "r");
        table_convergence.set_tex_format("It_{XFEM}", "r");
        table_convergence.set_tex_format("XFEM-time", "r");

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

//         xmodel.compute_interpolated_exact(exact_solution);
//         xmodel.output_results(cycle);
//         exact.output_distributed_solution(xmodel.get_output_triangulation(), cycle);
//         xmodel.output_distributed_solution(model.get_triangulation(),cycle, 1);
//         exact.output_distributed_solution(model.get_triangulation(), cycle);

        delete rhs_function;
        delete exact_solution;
        delete well;
}

void test_square_sin_7_rhow(std::string output_dir)
{
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH SIN(x) 7 RHO_W ::::::::::::::::\n\n" << std::endl;
    
    std::vector<double> excenter ({0.0, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.015, 0.03, 0.06, 0.1, 0.11, 0.120, 0.123, 0.125});

    std::vector<double> well_parameter ({-0.688174846114054,  -0.679917619489053,  -0.675789377751973,  -0.671661581890416,  -0.663407922173293,
        -0.655157828850421, -0.646912489920981,  -0.626327732672823, -0.564981242792659,  -0.445768346874862,  -0.299633782451967,  -0.266277654458329,
  -0.234439900760358,  -0.225201937764184,  -0.219126445875340});

    //--------------------------END SETTING----------------------------------
    
    ConvergenceTable table_convergence;
    
    unsigned int n_cycles = excenter.size();
    for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
        test_square_sin_7_rhow_func(false, cycle, excenter[cycle], well_parameter[cycle], table_convergence, output_dir);
    } 
    
    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ON SQUARE WITH SIN(x) 7 RHO_W - DONE ::::::::::::::::\n\n" << std::endl;
}

void test_multiple_wells(std::string output_dir)
{
  std::cout << "\n\n:::::::::::::::: MULTIPLE WELLS TEST ::::::::::::::::\n\n" << std::endl;
  
  //------------------------------SETTING----------------------------------
  std::string test_name = "multiple_";
  bool //fem = false, 
       fem_create = true;
  
  double p_a = 15.0,    //area of the model
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
  
  wells.push_back( new Well( well_radius,
                             Point<2>(-10.0,-10.0),
                             perm2fer, 
                             perm2tard));
  
  wells.push_back( new Well( well_radius,
                             Point<2>(0.0,0.0), 
                             perm2fer, 
                             perm2tard));
    
  wells.push_back( new Well( well_radius,
                             Point<2>(-10.0,8.0), 
                             perm2fer, 
                             perm2tard));
    
  wells.push_back( new Well( well_radius,
                             Point<2>(8.0,9.0), 
                             perm2fer, 
                             perm2tard));
    
  wells.push_back( new Well( well_radius,
                             Point<2>(5.0,-10.0), 
                             perm2fer, 
                             perm2tard));
    
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
  xmodel.set_name(test_name + "sgfem");
  xmodel.set_enrichment_method(Enrichment_method::sgfem);
//   xmodel.set_name(test_name + "xfem_shift");
//   xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
  
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
  
//   if(fem)
//   {
    model_fem.set_grid_create_type(ModelBase::load);
    //model_fem.set_computational_mesh(coarse_file, ref_flags_fine);
    model_fem.set_computational_mesh(coarse_file);
    model_fem.run();
    double l2_norm_fem = compare::L2_norm( model_fem.get_solution(),
                                            model_fem.get_triangulation()
                                            );
    
    std::cout << "l2 norm of fem solution: "  << l2_norm_fem << std::endl;
    return;
//   }
  
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
      
      l2_norm_dif = compare::L2_norm_diff( model_fem.get_solution(),
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
 
 
void test_multiple_wells2(std::string output_dir)
{
  std::cout << "\n\n:::::::::::::::: MULTIPLE WELLS TEST 2 ::::::::::::::::\n\n" << std::endl;
  
  //------------------------------SETTING----------------------------------
  std::string test_name = "multiple_2_";
  bool fem = true, 
       fem_create = false;
  
  double p_a = 15.0,    //area of the model
         well_radius = 0.02,
//          perm2fer = Parameters::perm2fer, 
//          perm2tard = Parameters::perm2tard,
         transmisivity = Parameters::transmisivity,
         enrichment_radius = 3.0;
         
  unsigned int n_well_q_points = 500;
         
  unsigned int n_aquifers = 2;
  std::vector<double> transmisivity_vec = {0.001, 0.1};
  // top / aq1 / aq2
  std::vector<double> perm2fer_1 = {0.1, 1e7, 0};
  std::vector<double> perm2fer_2 = {1e6, 0, 0};
  std::vector<double> perm2fer_3 = {1e1, 1e8, 0};
  std::vector<double> perm2fer_4 = {1e6, 1e4, 0};
  std::vector<double> perm2fer_5 = {1e6, 1e4, 0};
  
  std::vector<double> perm2tard_1 = {1e10, 1e10, 1e10};
 std::vector<double> perm2tard_2 = {1e10, 1e10, 0};
  
  
  //std::string input_dir = "../input/square_convergence/";
  std::string input_dir = "../output/multiple_2_fem/";
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
  
  wells.push_back( new Well( well_radius,
                             Point<2>(-10.0,-10.0)));
  wells.back()->set_perm2aquifer(perm2fer_1);
  wells.back()->set_perm2aquitard(perm2tard_1);
  
  wells.push_back( new Well( well_radius,
                             Point<2>(0.0,0.0)));
  wells.back()->set_perm2aquifer(perm2fer_2);
  wells.back()->set_perm2aquitard(perm2tard_1);
  
  wells.push_back( new Well( well_radius,
                             Point<2>(-10.0,8.0)));
  wells.back()->set_perm2aquifer(perm2fer_3);
  wells.back()->set_perm2aquitard(perm2tard_1);
  
  wells.push_back( new Well( well_radius,
                             Point<2>(8.0,9.0)));
  wells.back()->set_perm2aquifer(perm2fer_4);
  wells.back()->set_perm2aquitard(perm2tard_2);
  
  wells.push_back( new Well( well_radius,
                             Point<2>(5.0,-10.0)));
  wells.back()->set_perm2aquifer(perm2fer_5);
  wells.back()->set_perm2aquitard(perm2tard_1);

  
  //setting BC - pressure at the top of the wells
  wells[0]->set_pressure(-5*Parameters::pressure_at_top);
  wells[1]->set_pressure(-1*Parameters::pressure_at_top);
  wells[2]->set_pressure(12*Parameters::pressure_at_top);
  wells[3]->set_pressure(Parameters::pressure_at_top);
  wells[4]->set_pressure(10*Parameters::pressure_at_top);
  
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
  
  XModel xmodel(wells,"",n_aquifers);  
  xmodel.set_name(test_name + "sgfem");
  xmodel.set_enrichment_method(Enrichment_method::sgfem);
//   xmodel.set_name(test_name + "xfem_shift");
//   xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
  
  xmodel.set_output_dir(output_dir);
  xmodel.set_area(down_left,up_right);
  xmodel.set_transmisivity(transmisivity,0);
  xmodel.set_initial_refinement(4);                                     
  xmodel.set_enrichment_radius(enrichment_radius);
  xmodel.set_grid_create_type(ModelBase::rect);
  //xmodel.set_dirichlet_function(dirichlet);
  xmodel.set_adaptivity(true);
  xmodel.set_output_options(ModelBase::output_vtk_mesh);
  //xmodel.set_well_computation_type(Well_computation::sources);

  
  
  if(fem_create)
  {
    model_fem.set_area(down_left,up_right);
    model_fem.set_initial_refinement(5); 
    model_fem.set_grid_create_type(ModelBase::rect);
    model_fem.set_ref_coarse_percentage(0.3,0.05);
    model_fem.set_adaptivity(true);
    for (unsigned int cycle=0; cycle < 8; ++cycle)
    { 
      std::cout << "===== Model running   " << cycle << "   =====" << std::endl;
      model_fem.run (cycle);
      model_fem.output_results (cycle);
      std::cout << "===== Model finished =====" << std::endl;
    }
    return;
  }
  
  if(fem)
  {
    model_fem.set_grid_create_type(ModelBase::load);
    //model_fem.set_computational_mesh(coarse_file, ref_flags_fine);
    model_fem.set_computational_mesh(coarse_file);
    model_fem.run();
    double l2_norm_fem = compare::L2_norm( model_fem.get_solution(),
                                            model_fem.get_triangulation()
                                            );
    
    std::cout << "l2 norm of fem solution: "  << l2_norm_fem << std::endl;
//     return;
  }
  
  unsigned int n_cycles = 1;
  double l2_norm_dif;
  
  TableHandler table;
  
  for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
      table.add_value("Cycle",cycle);
 
      std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
      xmodel.run (cycle);  
      xmodel.output_results(cycle);
      xmodel.output_distributed_solution(model_fem.get_triangulation(),cycle,1);
      xmodel.output_distributed_solution(model_fem.get_triangulation(),cycle,2);
      std::cout << "===== XModel_simple finished =====" << std::endl;
      
      l2_norm_dif = compare::L2_norm_diff( model_fem.get_solution(),
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
  std::cout << "\n\n:::::::::::::::: MULTIPLE WELLS TEST 2 END ::::::::::::::::\n\n" << std::endl;
}

void test_output(std::string output_dir)
{
  std::string test_name = "test_output_";
  double p_a = 10.0,    //area of the model
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
  Function<2> *dirichlet = new compare::ExactSolution(well,radius);
  
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
  
  compare::ExactBase* exact_solution = new compare::ExactSolution(well, radius);
  double l2_norm_dif_xfem = compare::L2_norm_diff(xmodel.get_distributed_solution(),
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
  compare::ExactWellBase* exact_solution = new compare::ExactSolution(well, radius);
  //Function<2> *dirichlet_square = new compare::ExactSolution(well,radius);
  
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
  
  
  std::pair<double,double> l2_norm_dif_xfem;
 
      std::cout << "===== XModel_simple running   =====" << std::endl;
      
      xmodel.test_method(exact_solution);  
      
      
      std::cout << "===== XModel_simple finished =====" << std::endl;
      

      Vector<double> diff_vector;
      l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, exact_solution);
      
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

  
  
  XQuadratureCell * xquadrature = new XQuadratureCell(xdata, 
                                                      fe_values.get_mapping(),
                                                      XQuadratureCell::Refinement::edge
                                                     );
  xquadrature->refine(10);
  xquadrature->gnuplot_refinement(output_dir,true, true);
  
  Adaptive_integration adapt(xdata, fe,xquadrature,0);
  
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


// void test_adaptive_integration2(std::string output_dir)
// {
//     output_dir += "test_adaptive_integration_2/";
//   
//     unsigned int start_level = 5,
//                fine_level = 12;
//                
//     double p_a = 2.0,    //area of the model
//            well_radius = 1.0,
//            excenter = 0;
//          
//     Point<2> well_center(0+excenter,0+excenter);
//   
//     //--------------------------END SETTING----------------------------------
//   
//     Point<2> down_left(-p_a,-p_a);
//     Point<2> up_right(p_a, p_a);
//     std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
//   
//   
//     Well *well = new Well( well_radius,
//                            well_center);
//     well->set_pressure(1.0);
//     well->set_perm2aquifer(0,0);
//     well->set_perm2aquitard({0,0});
//     well->evaluate_q_points(100);
//     
//     Triangulation<2> tria;
//     GridGenerator::hyper_rectangle<2>(tria,down_left, up_right);
//     DBGMSG("tria size: %d\n",tria.n_active_cells());
//     
//     FE_Q<2> fe(1);
//     QGauss<2> quad(1);
//     FEValues<2> fe_values(fe, quad, UpdateFlags::update_default);
//     DoFHandler<2> dof_handler;
//     dof_handler.initialize(tria, fe);
//     
//     DoFHandler<2>::active_cell_iterator cell = dof_handler.begin_active();  
//     
//     std::vector<unsigned int> enriched_dofs(fe.dofs_per_cell);
//     enriched_dofs[0] = 4;
//     enriched_dofs[1] = 5;
//     enriched_dofs[2] = 6;
//     enriched_dofs[3] = 7;
//     std::vector<unsigned int> weights(fe.dofs_per_cell,0);
//     
//     std::vector<const Point<2>* > points (well->q_points().size());
//     //std::vector<const Point<2>* > points2 (well2->q_points().size());
//     for(unsigned int p=0; p < well->q_points().size(); p++)
//     {
//             points[p] = &(well->q_points()[p]);
//             //points2[p] = &(well2->q_points()[p]);
//     }
//     DBGMSG("N quadrature points: %d\n",points.size());
//     XDataCell* xdata = new XDataCell(cell, well,0,enriched_dofs, weights, points);
//     //xdata->add_data(well,1,enriched_dofs,weights,points2);
//     
//     cell->set_user_pointer(xdata);
//     fe_values.reinit(cell);
// 
//     TestIntegration_r2* func = new TestIntegration_r2(well);
//     
//     ConvergenceTable table_convergence;
//             
//     for(unsigned int i=0; i < fine_level-start_level-1; i++)
//     {
//         Adaptive_integration adapt(cell, fe,fe_values.get_mapping(),0);
//            
//         // compute integral of the function on the well edge
//         for(unsigned int j=0; j < start_level+i; j++)
//             adapt.refine_edge();
//   
//     //     adapt.gnuplot_refinement(output_dir,true, true);
//         // test, fine
//         std::pair<double, double> integrals = adapt.test_integration_2(func, fine_level-start_level-i);
//     
//         double difference = integrals.second-integrals.first;
//         std::cout << setprecision(16) << difference << std::endl;
//         
//         table_convergence.add_value("Refinement level",start_level+i);
//         table_convergence.set_tex_format("Refinement level", "r");
//       
//         table_convergence.add_value("difference",difference);
//         table_convergence.set_precision("difference", 3);
//         table_convergence.set_scientific("difference",true);
//   
//         table_convergence.evaluate_convergence_rates("difference", ConvergenceTable::reduction_rate);
//         table_convergence.evaluate_convergence_rates("difference", ConvergenceTable::reduction_rate_log2);
//         table_convergence.write_text(std::cout);
//     }
//     table_convergence.write_text(std::cout);
//     std::ofstream out_file;
//     out_file.open(output_dir + "convergence.tex");
//     table_convergence.write_tex(out_file);
//     out_file.close();
//         
//     out_file.open(output_dir + "convergence.txt");
//     table_convergence.write_text(out_file, 
//                                  TableHandler::TextOutputFormat::table_with_separate_column_description);
//     out_file.close();
// }


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
        
        TableHandler table;
        for(unsigned int l=refinement_level_min; l <= refinement_level_max; l++ )
        {
            std::cout << l << std::endl;
            unsigned int index = l - refinement_level_min;
            double rel_error = xmodel.test_adaptive_integration(well_characteristic_function,l);
            std::cout << "ERROR" << std::endl;
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
         well_radius = 0.02,
         enrichment_radius = 2.0,
         input_pressure = 5;
         
  unsigned int n_aquifers = 2, 
               initial_refinement = 5,
               n_well_q_points = 200;
  
  std::vector<double> transmisivity_vec = {0.001, 0.1};
  // top / aq1 / aq2
  std::vector<double> perm2fer_1 = {1e6, 1e4, 0};
  std::vector<double> perm2tard_1 = {1e10, 1e10, 1e10};
  std::vector<double> perm2fer_2 = {1e6, 1e4, 0};
  std::vector<double> perm2tard_2 = {1e10, 1e10, 1e10};
  
  //--------------------------END SETTING----------------------------------
  
  Point<2> down_left(-p_a,-p_a);
  Point<2> up_right(p_a, p_a);
  std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
  
  
  //vector of wells
  std::vector<Well*> wells;
  
  wells.push_back( new Well( well_radius,
                             Point<2>(-5.0,-1.1)));
  wells[0]->set_pressure(input_pressure);
  wells[0]->set_perm2aquifer(perm2fer_1);
  wells[0]->set_perm2aquitard(perm2tard_1);
  
  wells.push_back( new Well( well_radius,
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
  std::cout << "\n\n:::::::::::::::: TWO AQUIFERS TEST END ::::::::::::::::\n\n" << std::endl;
  
}


void test_enr_error(std::string output_dir)
{
    std::cout << "\n\n:::::::::::::::: TEST ENRICHMENT RADIUS ERROR ::::::::::::::::\n\n" << std::endl;
    
    //------------------------------SETTING----------------------------------
    double p_a = 10.0,    //area of the model
            excenter = 0,
            radius = p_a*std::sqrt(2),
            well_radius = 0.2,
            perm2fer = Parameters::perm2fer, 
            perm2tard = Parameters::perm2tard,
            transmisivity = 1.0,
            enrichment_radius = 50.0,
            well_pressure = 1.0;
            
    unsigned int n_well_q_points = 200;
            
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
    
    compare::ExactBase* exact_solution = new compare::ExactSolution(well, radius);

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
        xmodel.set_enrichment_radius(enrichment_radius);
        xmodel.set_grid_create_type(ModelBase::rect);
        xmodel.set_dirichlet_function(exact_solution);
        xmodel.set_adaptivity(true);
        xmodel.set_output_options(ModelBase::output_gmsh_mesh
                                //| ModelBase::output_solution
                                //| ModelBase::output_decomposed
                                | ModelBase::output_error);
        
        
        std::pair<double,double> l2_norm_dif_xfem;
    
        std::cout << "===== XModel_simple running   =====" << std::endl;
        xmodel.run();
        xmodel.test_enr_error();
        std::cout << "===== XModel_simple finished =====" << std::endl;
        
//         Vector<double> diff_vector;
//         l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, *exact_solution);
//         
//         xmodel.compute_interpolated_exact(exact_solution);
//         xmodel.output_results();
    }
    
    delete well;
    delete exact_solution;
    std::cout << "\n\n:::::::::::::::: TEST ENRICHMENT RADIUS ERROR - DONE ::::::::::::::::\n\n" << std::endl;
}


void test_wells_in_element(std::string output_dir)
{
    std::cout << "\n\n:::::::::::::::: WELLS IN ELEMENT TEST ::::::::::::::::\n\n" << std::endl;
    
    //------------------------------SETTING----------------------------------
    std::string test_name = "wells_in_element_";
    bool fem = true;
    
    double p_a = 2.0,    //area of the model
            well_radius = 0.02,
            well_pressure = 5,
            perm2fer = Parameters::perm2fer, 
            perm2tard = Parameters::perm2tard,
            transmisivity = Parameters::transmisivity,
            enrichment_radius = 0.4;
            
    unsigned int n_well_q_points = 100,
                refinement = 3;
            
    
    //--------------------------END SETTING----------------------------------
    
    Point<2> down_left(-p_a,-p_a);
    Point<2> up_right(p_a, p_a);
    std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
    
    
    //vector of wells
    std::vector<Well*> wells;
    
    wells.push_back( new Well( well_radius,
                               Point<2>(-0.375,-0.125)
                             )
                   );
    
    wells.push_back( new Well( Parameters::radius,
                               Point<2>(-0.125,-0.375)
                             )
                   );
        
    //   wells.push_back( new Well( Parameters::radius,
    //                              Point<2>(-10.0,8.0), 
    //                              Parameters::perm2fer, 
    //                              Parameters::perm2tard));
    //     
    //   wells.push_back( new Well( Parameters::radius,
    //                              Point<2>(8.0,9.0), 
    //                              Parameters::perm2fer, 
    //                              Parameters::perm2tard));
    //     
    //   wells.push_back( new Well( Parameters::radius,
    //                              Point<2>(5.0,-10.0), 
    //                              Parameters::perm2fer, 
    //                              Parameters::perm2tard));
        
    //setting BC - pressure at the top of the wells
//     wells[0]->set_pressure(well_pressure);
//     wells[1]->set_pressure(well_pressure);
    //   wells[2]->set_pressure(2*Parameters::pressure_at_top);
    //   wells[3]->set_pressure(Parameters::pressure_at_top);
    //   wells[4]->set_pressure(3*Parameters::pressure_at_top);
    
    for(unsigned int w=0; w < wells.size(); w++)
    {
        wells[w]->set_active();
        wells[w]->set_pressure(well_pressure);
        wells[w]->evaluate_q_points(n_well_q_points);
        wells[w]->set_perm2aquifer(0,perm2fer);
        wells[w]->set_perm2aquitard({perm2tard, 0.0});
    }


    //FEM model creation
    Model fem_model(wells);  
    fem_model.set_name(test_name + "fem");
    fem_model.set_output_dir(output_dir);
    fem_model.set_area(down_left,up_right);
    fem_model.set_transmisivity(transmisivity,0);
    fem_model.set_initial_refinement(refinement);  
    fem_model.set_ref_coarse_percentage(0.3,0.05);
    //fem_model.set_ref_coarse_percentage(0.3,0.05);
    
    fem_model.set_grid_create_type(ModelBase::rect);
    //fem_model.set_computational_mesh(coarse_file);
    compare::ExactSolutionZero* zero = new compare::ExactSolutionZero();
    fem_model.set_dirichlet_function(zero);
    fem_model.set_adaptivity(true);
    fem_model.set_output_options(ModelBase::output_gmsh_mesh
                                 | ModelBase::output_solution);
 

    XModel xmodel(wells);  
    xmodel.set_name(test_name + "sgfem");
    xmodel.set_enrichment_method(Enrichment_method::sgfem);
//       xmodel.set_name(test_name + "xfem_shift");
//       xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
//     xmodel.set_name(test_name + "xfem_ramp");
//     xmodel.set_enrichment_method(Enrichment_method::xfem_ramp);
//     xmodel.set_name(test_name + "xfem");
//     xmodel.set_enrichment_method(Enrichment_method::xfem);
    
    xmodel.set_output_dir(output_dir);
    xmodel.set_area(down_left,up_right);
    xmodel.set_transmisivity(transmisivity,0);
    xmodel.set_initial_refinement(refinement);                                     
    xmodel.set_enrichment_radius(enrichment_radius);
    xmodel.set_grid_create_type(ModelBase::rect);
    //xmodel.set_dirichlet_function(dirichlet);
    xmodel.set_adaptivity(true);
    //xmodel.set_well_computation_type(Well_computation::sources);
    xmodel.set_output_options(ModelBase::output_adaptive_plot);


    unsigned int n_fem_cycles = 8;
    std::vector<double> fem_l2_norm(n_fem_cycles,0),
                        fem_l2_diff(n_fem_cycles,0);;
    if(fem)
    {
        for (unsigned int cycle=0; cycle < n_fem_cycles; ++cycle)
        { 
            fem_model.run(cycle);
            fem_model.output_results (cycle);
            
            Vector<double> diff_vector;
            fem_model.integrate_difference(diff_vector, zero, false);
            
            fem_l2_norm[cycle] = diff_vector.l2_norm();
            
            if(cycle > 0) fem_l2_diff[cycle] = std::abs(fem_l2_norm[cycle-1] - fem_l2_norm[cycle]);
            
            for(auto &v : fem_l2_diff)
                std::cout << v << std::endl;
            
        }
    }
    
    unsigned int n_cycles = 1;
    double l2_norm_dif;
    
    TableHandler table;
    
    for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
        table.add_value("Cycle",cycle);
    
        std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
        
        xmodel.run (cycle);  
        xmodel.output_results(cycle);
        xmodel.output_distributed_solution(fem_model.get_triangulation(),cycle);
        std::cout << "===== XModel_simple finished =====" << std::endl;
      
        l2_norm_dif = compare::L2_norm_diff( fem_model.get_solution(),
                                                xmodel.get_distributed_solution(),
                                                fem_model.get_triangulation()
                                            );
        std::cout << l2_norm_dif <<std::endl;
//       
//       table.add_value("$\\|x_{XFEM}-x_{FEM}\\|_{L^2(\\Omega)}$",l2_norm_dif);
//       table.set_precision("$\\|x_{XFEM}-x_{FEM}\\|_{L^2(\\Omega)}$", 2);
//       table.set_scientific("$\\|x_{XFEM}-x_{FEM}\\|_{L^2(\\Omega)}$",true);
//       
//       table.add_value("dofs",xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
//       table.add_value("enriched dofs",xmodel.get_number_of_dofs().second);
//       table.add_value("Iterations",xmodel.solver_iterations());
// 
//       
//       //write the table every cycle (to have at least some results if program fails)
//       table.write_text(std::cout);
//       std::ofstream out_file;
//       out_file.open(output_dir + xmodel.name() + ".tex");
//       table.write_tex(out_file);
//       out_file.close();
    } 
  //*/
  
    for(unsigned int w=0; w < wells.size(); w++)
    {
        delete wells[w];
    }
    delete zero;
    std::cout << "\n\n:::::::::::::::: WELLS IN ELEMENT TEST END ::::::::::::::::\n\n" << std::endl;
}


void test_xquadrature_well(std::string output_dir)
{
    output_dir += "test_xquadrature_well/";
    double well_radius = 0.011,
           scale_width = 2,
           width = scale_width * well_radius,
           excenter = 0.5;
            
    unsigned int n_well_q_points = 500;
            
    Point<2> well_center(0+excenter,0+excenter);
    
    //--------------------------END SETTING----------------------------------

    
    Well *well = new Well( well_radius,
                           well_center,
                           0, 
                           0);
    well->set_pressure(well_radius);
    well->evaluate_q_points(n_well_q_points);
    
    XQuadratureWell xquad(well, width);
    xquad.refine(6);
    xquad.gnuplot_refinement(output_dir, true, false);
    
    double sum = 0;
    for(unsigned int i=0; i < xquad.size(); i++)
        sum += xquad.weight(i);
    
    std::cout << "Control sum of weights: " << sum
     << "\t" << scale_width*well_radius * 2*M_PI << std::endl;
    
    std::cout << "Test integration [0,pi/2] x [r,5r]: " << std::endl;
    double integral = 0;
    for(unsigned int q = 0; q < xquad.size(); q++)
    {
        double phi = xquad.polar_point(q)[1];
        if( (0 <= phi) && (phi <= M_PI/2))
            integral += xquad.polar_point(q)[0] * xquad.weight(q);
    }
    double analytic_integral = M_PI/4 * ((scale_width+1)*(scale_width+1) - 1) * well_radius * well_radius;
    std::cout << setprecision(16) << "numeric =  " << integral 
                                  << "\t analytic = " << analytic_integral
                                  << "\t error = " << integral - analytic_integral << std::endl;
    
    SmoothStep smooth_step(well, width);
    
    unsigned int n = 300;
    double dd = 0.1 / n,
           x = well_center[0];
           std::cout << setw(15) << "r" << "\t" << setw(10) << "s.value(p)" << "\t"
                  << setw(10) << "s.value(dis)" << std::endl;
    for(unsigned int i=0; i < n; i++)
    {
        Point<2> p(x,x);
        double dis = well_center.distance(p);
        std::cout << setprecision(16);
        std::cout << setw(15) << dis << "\t" << setw(10) << smooth_step.value(p) << "\t"
                  << setw(10) << smooth_step.value(dis) << std::endl;
        x += dd;
    }
    
    
    std::cout << ".....................test of quadratures.............." << std::endl;
    QGaussLogR<1> gauss_log_r0(12, Point<1>(0.0), 0.5, true);
    integral = 0;
    for(unsigned int q = 0; q < gauss_log_r0.size(); q++)
    {
        double r = gauss_log_r0.point(q)[0] * 2;
        integral += std::log(r) * 2 * gauss_log_r0.weight(q);//xquad_log.polar_point(q)[0] *
    }
    analytic_integral = 2*(log(2)-1);
    std::cout << setprecision(16) << "numeric =  " << integral 
                                  << "\t analytic = " << analytic_integral
                                  << "\t error = " << integral - analytic_integral << std::endl;
                                  
    
    QGaussLogR<1> gauss_log_r(12, Point<1>(0.0), 1/((scale_width+1)*well_radius), true);
    integral = 0;
    for(unsigned int q = 0; q < gauss_log_r.size(); q++)
    {
        double r = gauss_log_r.point(q)[0] * (width + well->radius());
        if( r > well_radius)
            integral += std::log(r) * width * gauss_log_r.weight(q);
//             integral += std::log(r) * width * gauss_log_r.weight(q);
    }
    std::cout << "gauss_log integral = " << integral << std::endl;
    
    
    //log
    QGauss<1> gauss(12);
    double integral_log = 0,
           integral_r = 0,
           integral_rr = 0;
    for(unsigned int q = 0; q < gauss.size(); q++)
    {
        double r = gauss.point(q)[0]* width + well->radius() ;
        integral_log += std::log(r) *  width * gauss.weight(q);
        integral_r += 1/r *  width * gauss.weight(q);
        integral_rr += 1/(r*r) *  width * gauss.weight(q);
        
    }
    double analytic_integral_log = (
                                 (scale_width+1)*well_radius*(log((scale_width+1)*well_radius)-1) 
                                 - well_radius*(log(well_radius)-1)
                                 ),
           analytic_integral_r = std::log(scale_width+1),
           analytic_integral_rr = -1/((scale_width+1)*well_radius) + 1/well_radius;
    std::cout << setprecision(16) << "numeric =  " << integral_log
                                  << "\t analytic = " << analytic_integral_log
                                  << "\t error = " << integral_log - analytic_integral_log << std::endl;
    std::cout << setprecision(16) << "numeric =  " << integral_r
                                  << "\t analytic = " << analytic_integral_r
                                  << "\t error = " << integral_r - analytic_integral_r << std::endl;
    std::cout << setprecision(16) << "numeric =  " << integral_rr
                                  << "\t analytic = " << analytic_integral_rr
                                  << "\t error = " << integral_rr - analytic_integral_rr << std::endl;
                                  
    
    XQuadratureWellLog xquad_log(well, width, 500,10);
    xquad_log.refine(0);
    xquad_log.gnuplot_refinement(output_dir, true, false);
    
    sum = 0;
    for(unsigned int i=0; i < xquad_log.size(); i++)
        sum += xquad_log.weight(i);
    
    std::cout << "Control sum of weights: " << sum
     << "\t" << scale_width*well_radius * 2*M_PI << std::endl;
    
//     for(unsigned int q = 0; q < xquad_log.size(); q++)
//     {
//         std::cout << xquad_log.weight(q) << std::endl;
//     }
    
    std::cout << "Test integration [0,pi/2] x [r,5r]: " << std::endl;
    integral_log = integral_r = integral_rr = 0;
    for(unsigned int q = 0; q < xquad_log.size(); q++)
    {
        double phi = xquad_log.polar_point(q)[1];
        double r = xquad_log.polar_point(q)[0];
        if( (0 <= phi) && (phi <= M_PI/2))
        {
            integral_log += std::log(r) * xquad_log.weight(q);
            integral_r += 1/r * xquad_log.weight(q);
            integral_rr += 1/(r*r) * xquad_log.weight(q);
        }
    }
    analytic_integral_log *= M_PI/2;
    analytic_integral_r *= M_PI/2;
    analytic_integral_rr *= M_PI/2;
    std::cout << setprecision(16) << "numeric =  " << integral_log
                                  << "\t analytic = " << analytic_integral_log
                                  << "\t error = " << integral_log - analytic_integral_log << std::endl;
    std::cout << setprecision(16) << "numeric =  " << integral_r
                                  << "\t analytic = " << analytic_integral_r
                                  << "\t error = " << integral_r - analytic_integral_r << std::endl;
    std::cout << setprecision(16) << "numeric =  " << integral_rr
                                  << "\t analytic = " << analytic_integral_rr
                                  << "\t error = " << integral_rr - analytic_integral_rr << std::endl;
                                  
    delete well;
}

void test_xquadrature_well_2(std::string output_dir)
{
    output_dir += "test_xquadrature_well_2/";
    double well_radius = 0.011,
           excenter = 0.5;
            
    unsigned int n_well_q_points = 500;
            
    Point<2> well_center(0+excenter,0+excenter);
    
    //--------------------------END SETTING----------------------------------

    
    Well *well = new Well( well_radius,
                           well_center,
                           0, 
                           0);
    well->set_pressure(well_radius);
    well->evaluate_q_points(n_well_q_points);
    
    
    std::vector<double> ex(20);
    for(unsigned int i = 0; i < 20; i++)
    {
        double scale_width = 0.5 + 0.1*i,
               width = scale_width * well_radius;
               
        double analytic_integral_rr = -1/((scale_width+1)*well_radius) + 1/well_radius;
           
     
        XQuadratureWellLog xquad_log(well, width,500, 10);
        xquad_log.refine(0);
//         xquad_log.gnuplot_refinement(output_dir, true, false);
    
        
//         std::cout << "Test integration [0,pi/2] x [r,a*r]: " << std::endl;
        double integral_rr = 0;
        for(unsigned int q = 0; q < xquad_log.size(); q++)
        {
            double phi = xquad_log.polar_point(q)[1];
            double r = xquad_log.polar_point(q)[0];
            if( (0 <= phi) && (phi <= M_PI/2))
            {
                integral_rr += 1/(r*r) * xquad_log.weight(q);
            }
        }
        analytic_integral_rr *= M_PI/2;

//         std::cout << setprecision(16) << "numeric =  " << integral_rr
//                                     << "\t analytic = " << analytic_integral_rr
//                                     << "\t error = " << integral_rr - analytic_integral_rr << std::endl;
                                    
        ex[i] = (integral_rr - analytic_integral_rr) / M_PI * 2;
        std::cout << setprecision(16) << "ex =  " << ex[i] << std::endl;
    }
                                  
    delete well;
}

void test_xquadrature_well_band(std::string output_dir)
{
    output_dir += "test_xquadrature_well_band/";
    double well_radius = 0.011,
           scale_width = 1,
           width = scale_width * well_radius,
           excenter = 0.5;
            
    unsigned int n_well_q_points = 500;
            
    Point<2> well_center(0+excenter,0+excenter);
    
    //--------------------------END SETTING----------------------------------

    
    Well *well = new Well( well_radius,
                           well_center,
                           0, 
                           0);
    well->set_pressure(well_radius);
    well->evaluate_q_points(n_well_q_points);
    
    
    
    XQuadratureWellBand xquad(well, width, 7);
    xquad.refine(1);
    xquad.gnuplot_refinement(output_dir, true, false);
    
    double sum = 0;
    for(unsigned int i=0; i < xquad.size(); i++)
        sum += xquad.weight(i);
    
    std::cout << "Control sum of weights: " << sum
     << "\t" << scale_width*well_radius * 2*M_PI << std::endl;
    

    
    std::cout << "Test integration [0,pi/2] x [r,5r]: " << std::endl;
    double analytic_integral_log = (
                                 (scale_width+1)*well_radius*(log((scale_width+1)*well_radius)-1) 
                                 - well_radius*(log(well_radius)-1)
                                 ),
           analytic_integral_r = std::log(scale_width+1),
           analytic_integral_rr = -1/((scale_width+1)*well_radius) + 1/well_radius;
           
    double integral_log = 0,
           integral_r = 0,
           integral_rr = 0;
    for(unsigned int q = 0; q < xquad.size(); q++)
    {
        double phi = xquad.polar_point(q)[1];
        double r = xquad.polar_point(q)[0];
        if( (0 <= phi) && (phi <= M_PI/2))
        {
            integral_log += std::log(r) * xquad.weight(q);
            integral_r += 1/r * xquad.weight(q);
            integral_rr += 1/(r*r) * xquad.weight(q);
        }
    }
    analytic_integral_log *= M_PI/2;
    analytic_integral_r *= M_PI/2;
    analytic_integral_rr *= M_PI/2;
    std::cout << setprecision(16) << "numeric =  " << integral_log
                                  << "\t analytic = " << analytic_integral_log
                                  << "\t error = " << integral_log - analytic_integral_log << std::endl;
    std::cout << setprecision(16) << "numeric =  " << integral_r
                                  << "\t analytic = " << analytic_integral_r
                                  << "\t error = " << integral_r - analytic_integral_r << std::endl;
    std::cout << setprecision(16) << "numeric =  " << integral_rr
                                  << "\t analytic = " << analytic_integral_rr
                                  << "\t error = " << integral_rr - analytic_integral_rr << std::endl;
                                  
    delete well;
}


void visualize_source_term()
{
    Triangulation<2> tria;
    string grid_file = "/home/pavel/xfem_project/003/output/xfem_mesh_6.msh";
    
    
    //open filestream with mesh from GMSH
    std::ifstream in;
    GridIn<2> gridin;
    in.open(grid_file);
    //attaching object of triangulation
    gridin.attach_triangulation(tria);
    if(in.is_open())
    {
      //reading data from filestream
      gridin.read_msh(in);
    }          
    else
    {
      xprintf(Err, "Could not open coarse grid file: %s\n", grid_file.c_str());
    }
    
    double p_a = 2.0,    //area of the model
           excenter = 0.004,
           radius = p_a*std::sqrt(2),
           well_radius = 0.003,
           perm2fer = 5e1,
           perm2tard = 1e3,
           k_wave_num = 1.5,
           amplitude = 0.5,
           well_pressure = 9;
         
    unsigned int n_well_q_points = 200;
            
    Point<2> well_center(0+excenter,0+excenter);
    //--------------------------END SETTING----------------------------------
    
    Well well(well_radius, well_center);
    well.set_pressure(well_pressure);
    well.set_perm2aquifer(0,perm2fer);
    well.set_perm2aquitard({perm2tard, 0.0});
    well.evaluate_q_points(n_well_q_points);

    compare::ExactSolution4 exact_solution(&well, radius, k_wave_num, amplitude);
    compare::ExactBase* source = new compare::Source4(exact_solution);
    ExactModel exact(source);
    exact.output_distributed_solution(tria);
    
    delete source;
}


void test_polar_radius_sin_3(std::string output_dir)
{
    std::cout << "\n\n:::::::::::::::: POLAR RADIUS TEST ON SQUARE WITH SIN(x) 3 ::::::::::::::::\n\n" << std::endl;
    //------------------------------SETTING----------------------------------
    std::string test_name = "polar_test_sin_3_";
  
    double p_a = 2.0,    //area of the model
           excenter = 0.0,//0.004,
           radius = p_a*std::sqrt(2),
           well_radius = 0.007,
           perm2fer = 1e5,
           perm2tard = 1e10,
           transmisivity = Parameters::transmisivity,
           k_wave_num = 1.5,
           amplitude = 0.5,
           well_pressure = 9,
           enrichment_radius = 0.3;
         
    unsigned int n_well_q_points = 300,
                 initial_refinement = 9;
            
    Point<2> well_center(0+excenter,0+excenter);
    //--------------------------END SETTING----------------------------------
  
    Point<2> down_left(-p_a,-p_a);
    Point<2> up_right(p_a, p_a);
    std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
    
    Well *well = new Well( well_radius,
                            well_center);
    well->set_pressure(well_pressure);
    well->set_perm2aquifer(0,perm2fer);
    well->set_perm2aquitard({perm2tard, 0.0});
    well->evaluate_q_points(n_well_q_points);
        
    compare::ExactSolution4 *exact_solution = new compare::ExactSolution4(well, radius, k_wave_num, amplitude);
    Function<2> *dirichlet_square = exact_solution;
    Function<2> *rhs_function = new compare::Source4(*exact_solution);
    
    XModel_simple xmodel(well);  
//     xmodel.set_name(test_name + "sgfem");
//     xmodel.set_enrichment_method(Enrichment_method::sgfem);
    xmodel.set_name(test_name + "xfem_shift");
    xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
//       xmodel.set_name(test_name + "xfem_ramp");
//       xmodel.set_enrichment_method(Enrichment_method::xfem_ramp);
//       xmodel.set_name(test_name + "xfem");
//       xmodel.set_enrichment_method(Enrichment_method::xfem);
    
    xmodel.set_output_dir(output_dir);
    xmodel.set_area(down_left,up_right);
    xmodel.set_transmisivity(transmisivity,0);
    xmodel.set_initial_refinement(initial_refinement);                                     
    xmodel.set_enrichment_radius(enrichment_radius);
    xmodel.set_grid_create_type(ModelBase::rect);
    xmodel.set_dirichlet_function(dirichlet_square);
    xmodel.set_rhs_function(rhs_function);
    xmodel.set_adaptivity(false);
    xmodel.set_output_options(ModelBase::output_gmsh_mesh
//                               | ModelBase::output_solution
//                               | ModelBase::output_decomposed
                            | ModelBase::output_adaptive_plot
                            | ModelBase::output_error
//                             | ModelBase::output_matrix
                             );
    
    ExactModel exact(exact_solution);

    unsigned int n_cycles = 30;
    std::pair<double,double> l2_norm_dif_xfem;
    
    ConvergenceTable table_convergence;
    
    double polar_radius = 0.1;
    
    for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
        table_convergence.add_value("Cycle",cycle);
        table_convergence.set_tex_format("Cycle", "r");
        table_convergence.add_value("g",polar_radius);
        table_convergence.set_tex_format("g", "r");
        
        std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
        
        xmodel.set_well_band_width_ratio(polar_radius);
        xmodel.run (cycle);  
        xmodel.output_results(cycle);
        
        std::cout << "===== XModel_simple finished =====" << std::endl;
      
        Vector<double> diff_vector;
        l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, exact_solution);
        
        table_convergence.add_value("X_L2",l2_norm_dif_xfem.second);
        table_convergence.set_tex_caption("X_L2","$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$");
        table_convergence.set_precision("X_L2", 4);
        table_convergence.set_scientific("X_L2",true);
      
        table_convergence.add_value("XFEM-dofs",
                                    xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
        table_convergence.add_value("XFEM-enriched dofs",xmodel.get_number_of_dofs().second);
        table_convergence.add_value("It_{XFEM}",xmodel.solver_iterations());
        
        table_convergence.add_value("XFEM-time",xmodel.last_run_time());
        table_convergence.set_precision("XFEM-time", 3);
        
        table_convergence.set_tex_format("XFEM-dofs", "r");
        table_convergence.set_tex_format("XFEM-enriched dofs", "r");
        table_convergence.set_tex_format("It_{XFEM}", "r");
        table_convergence.set_tex_format("XFEM-time", "r");
      
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
        
        xmodel.compute_interpolated_exact(exact_solution);
        xmodel.output_results(cycle);
//         exact.output_distributed_solution(xmodel.get_output_triangulation(), cycle);
        
        polar_radius += 0.03;
    } 
}

void test_radius_convergence_05(std::string output_dir)
{
    std::cout << "\n\n:::::::::::::::: RADIUS CONVERGENCE TEST ON SQUARE WITH 05 ::::::::::::::::\n\n" << std::endl;
    
    //------------------------------SETTING----------------------------------
    std::string test_name = "radius_convergence_05_";
    bool fem = true;   //compute h1 norm of the error of the regular part ur
    
//     double p_a = 2.0,    //area of the model
//            excenter = 0.004,
//            well_radius = 0.003,
//            perm2fer = 1e5,
//            perm2tard = 1e10,
//            
//            transmisivity = 1.0,
//            k_wave_num = 6,
//            amplitude = 4,
//            well_pressure = 4.0;
//          
//     unsigned int n_well_q_points = 200,
//                  initial_refinement = 3;
//             
//     Point<2> well_center(0+excenter,0+excenter);
//     
//     double well_parameter = -0.671661581890416;

    double p_a = 2.0,    //area of the model
           excenter = 0.004,
           well_radius = 0.003,
           perm2fer = 1e5,
           perm2tard = 1e10,
           
           transmisivity = 1.0,
           amplitude = 0.7,
           well_pressure = 4.0;
         
    unsigned int n_well_q_points = 200,
                 initial_refinement = 3;
            
    Point<2> well_center(0+excenter,0+excenter);
    
    double well_parameter = -0.688173755012835;
    
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
    
//     compare::ExactSolution7 *exact_solution = new compare::ExactSolution7(well, k_wave_num, amplitude);
//     Function<2> *dirichlet_square = exact_solution;
//     Function<2> *rhs_function = new compare::Source7(*exact_solution);
    compare::ExactSolution5 *exact_solution = new compare::ExactSolution5(well, amplitude);
    exact_solution->set_well_parameter(well_parameter);
    Function<2> *rhs_function = new compare::Source5(*exact_solution);
  
    if (fem)    //compute h1 norm of the error of the regular part ur
    {
        //   FEM model creation
        Well *well_fem = new Well(well);
//         well_fem->set_pressure(0);
        well_fem->evaluate_q_points(n_well_q_points);
        well_fem->set_inactive();
        
//         compare::ExactSolution7 *exact_solution_fem = new compare::ExactSolution7(well_fem, k_wave_num, amplitude);
//         Function<2> *rhs_function_fem = new compare::Source7(*exact_solution_fem);
        compare::ExactSolution5 *exact_solution_fem = new compare::ExactSolution5(well_fem, amplitude);
        Function<2> *rhs_function_fem = new compare::Source5(*exact_solution_fem);
        
        Model_simple model(well_fem);  
        model.set_name(test_name + "fem");
        model.set_output_dir(output_dir);
        model.set_area(down_left,up_right);
        model.set_transmisivity(transmisivity,0);
        model.set_initial_refinement(initial_refinement);  
        model.set_ref_coarse_percentage(1.0,0.0);
        model.set_grid_create_type(ModelBase::rect);
        model.set_dirichlet_function(exact_solution_fem);
        model.set_rhs_function(rhs_function_fem);
        model.set_adaptivity(true);
        model.set_output_options(ModelBase::output_gmsh_mesh
                                | ModelBase::output_solution
                                | ModelBase::output_error);
        
        unsigned int n_cycles = 8;
        std::vector<double> fem_errors(n_cycles);
        for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
        { 
            model.run(cycle);
            model.output_results(cycle);
            Vector<double> diff_vector;
            std::pair<double,double> h1_norm = model.integrate_difference(diff_vector, exact_solution_fem, false);
            fem_errors[cycle] = h1_norm.second;
            std::cout << "FEM error = " << h1_norm.second << std::endl;
        }
        //write the errors down:
        std::cout << "FEM errors: " << std::endl;
        for (unsigned int i=0; i < n_cycles; ++i)
        { 
            double h = 2*p_a / pow(2.0,initial_refinement+i);
            std::cout << h << "\t" << fem_errors[i] << std::endl;
        }
        delete well_fem;
        delete rhs_function_fem;
        delete exact_solution_fem;
        return;
    }
  
    std::vector<double> enrichment_radius;
    for(unsigned int r=0; r < 6; r++)
    {
        double val = pow(2.0,r+4)*well_radius;
//         double val = 51.2;
        enrichment_radius.push_back(val);
        std::cout << val << endl;
    }
    enrichment_radius[5] = 1.0;
//     return;
  
    ConvergenceTable table_convergence;
    for(unsigned int r=0; r < enrichment_radius.size(); r++)
    {
        stringstream name_error, name_edofs;
        name_error << "X_L2_" << r;
        name_edofs << "edofs" << r;
        
        XModel_simple xmodel(well, ""); 
        //     xmodel.set_name(test_name + "xfem");
        //     xmodel.set_enrichment_method(Enrichment_method::xfem);
        //     xmodel.set_name(test_name + "xfem_ramp");
        //     xmodel.set_enrichment_method(Enrichment_method::xfem_ramp);
//             xmodel.set_name(test_name + "xfem_shift");
//             xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
        xmodel.set_name(test_name + "sgfem"); 
        xmodel.set_enrichment_method(Enrichment_method::sgfem);
        
        xmodel.set_output_dir(output_dir);
        xmodel.set_area(down_left,up_right);
        xmodel.set_transmisivity(transmisivity,0);
        xmodel.set_initial_refinement(initial_refinement);                                     
        xmodel.set_enrichment_radius(enrichment_radius[r]);
        xmodel.set_grid_create_type(ModelBase::rect);
        xmodel.set_dirichlet_function(exact_solution);
        xmodel.set_rhs_function(rhs_function);
        xmodel.set_adaptivity(true);
        xmodel.set_output_options(ModelBase::output_gmsh_mesh
//                                   | ModelBase::output_solution
//                                   | ModelBase::output_decomposed
        //                           | ModelBase::output_adaptive_plot
                                | ModelBase::output_error);
        
        unsigned int n_cycles = 7;
        std::cout << "Cycles: " << n_cycles << std::endl;
        std::pair<double,double> l2_norm_dif_fem, l2_norm_dif_xfem;
    
        for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
        { 
            if(r==0)
            {
                double h = 2*p_a / pow(2.0,initial_refinement+cycle);
                table_convergence.add_value("h",h);
                table_convergence.set_tex_format("h", "r");
            }
        
            std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
        
            xmodel.run (cycle);  
        
            //xmodel.output_distributed_solution(*fine_triangulation,cycle);
            std::cout << "===== XModel_simple finished =====" << std::endl;
    
            Vector<double> diff_vector;
            l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, exact_solution);
            
            table_convergence.add_value(name_error.str(),l2_norm_dif_xfem.second);
            table_convergence.set_tex_caption(name_error.str(),"$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$");
            table_convergence.set_precision(name_error.str(), 2);
            table_convergence.set_scientific(name_error.str(),true);
            
            table_convergence.add_value(name_edofs.str(),xmodel.get_number_of_dofs().second);
            table_convergence.set_tex_format(name_edofs.str(), "r");
        }   // for cycle
        
        //write the table every cycle (to have at least some results if program fails)
        table_convergence.write_text(std::cout);
        std::ofstream out_file;
        std::stringstream filename;
        filename << output_dir + xmodel.name();
        out_file.open(filename.str() + ".tex");
        table_convergence.write_tex(out_file);
        out_file.close();
        
        out_file.open(filename.str() + ".txt");
        table_convergence.write_text(out_file, 
                                    TableHandler::TextOutputFormat::table_with_separate_column_description);
        out_file.close();
    }   //for r
      
  delete well;
  delete rhs_function;
  delete exact_solution;
  
  std::cout << "\n\n:::::::::::::::: RADIUS CONVERGENCE TEST ON SQUARE WITH 05 - DONE ::::::::::::::::\n\n" << std::endl;
}


void test_two_wells(std::string output_dir)
{
  std::cout << "\n\n:::::::::::::::: TWO WELLS TEST ::::::::::::::::\n\n" << std::endl;
  bool fem = false;
  //------------------------------SETTING----------------------------------
//   std::string test_name = "two_wells_enr_";
  std::string test_name = "two_wells_";
  
  double p_a = 10.0,    //area of the model
         well_radius = 0.003,
//          enrichment_radius = 2.0,
         enrichment_radius = 0.6,
         well_pressure1 = 150,
         well_pressure2 = 100,
         
         k_wave_num = 1,
         amplitude = 80;
         
  unsigned int n_aquifers = 1, 
               initial_refinement = 3,
               n_well_q_points = 200;
  
//   double transmisivity = 1;
  double transmisivity = 1e-3;
  // top / aq1 / aq2
  double perm2fer_1 = 100;
  double perm2tard_1 = 1e10;
  double perm2fer_2 = 100;
  double perm2tard_2 = 1e10;
  
//   std::vector<double> well_parameters ({-25.1751953630960,  -18.0051409109026}); // T=1
  std::vector<double> well_parameters ({-41.3671204832712,  -30.5952544054495}); // T=1e-3
  
  //--------------------------END SETTING----------------------------------
  
  Point<2> down_left(0,0);
  Point<2> up_right(p_a, p_a);
  std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
    
  //vector of wells
  std::vector<Well*> wells;
  
  wells.push_back( new Well( well_radius,
                             Point<2>(4.1,4.3)));
  wells[0]->set_pressure(well_pressure1);
  wells[0]->set_perm2aquifer(0, perm2fer_1);
  wells[0]->set_perm2aquitard({perm2tard_1, 0});
  
  wells.push_back( new Well( well_radius,
                             Point<2>(5.7,5.9)));
  wells[1]->set_pressure(well_pressure2);
  wells[1]->set_perm2aquifer(0, perm2fer_2);
  wells[1]->set_perm2aquitard({perm2tard_2, 0});  
  
  for(unsigned int w=0; w < wells.size(); w++)
  {
    wells[w]->evaluate_q_points(n_well_q_points);
  }
  
    compare::ExactSolutionMultiple *exact_solution = new compare::ExactSolutionMultiple(k_wave_num, amplitude);
    exact_solution->set_wells(wells, well_parameters, well_parameters);
    Function<2> *rhs_function = new compare::SourceMultiple(transmisivity, *exact_solution);
    unsigned int n_cycles = 9;

    //FEM model creation
    if(fem) //computing H1 norm of regular part of solution
    {
        std::vector<Well*> fem_wells;
        fem_wells.push_back(new Well(wells[0]));
        fem_wells.push_back(new Well(wells[1]));
        
        fem_wells[0]->set_inactive();
        fem_wells[1]->set_inactive();
        fem_wells[0]->evaluate_q_points(n_well_q_points);
        fem_wells[1]->evaluate_q_points(n_well_q_points);

        compare::ExactSolutionMultiple *exact_solution_fem = new compare::ExactSolutionMultiple(k_wave_num, amplitude);
        exact_solution->set_wells(fem_wells, well_parameters, well_parameters);
        Function<2> *rhs_function_fem = new compare::SourceMultiple(transmisivity, *exact_solution_fem);
    
        Model model(fem_wells);    
//     model.set_name(test_name + "fem");
        model.set_name(test_name + "fem_reg");
        model.set_output_dir(output_dir);
        model.set_area(down_left,up_right);
        model.set_transmisivity(transmisivity,0);
        model.set_initial_refinement(initial_refinement);  
        model.set_ref_coarse_percentage(1.0,0.0);
//         model.set_ref_coarse_percentage(0.4,0.0);
        model.set_grid_create_type(ModelBase::rect);
        model.set_dirichlet_function(exact_solution_fem);
        model.set_rhs_function(rhs_function_fem);
        model.set_adaptivity(true);
        model.set_output_options(ModelBase::output_gmsh_mesh
                                | ModelBase::output_solution
                                | ModelBase::output_error);
    
//         model.run (0);
//         model.output_results (0);
        ExactModel exact_fem(exact_solution_fem);
        
        ConvergenceTable table_convergence_fem;
        
        for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
        { 
            table_convergence_fem.add_value("Cycle",cycle);
            table_convergence_fem.set_tex_format("Cycle", "r");
            double h = 2*p_a / pow(2.0,initial_refinement+cycle);
            table_convergence_fem.add_value("h",h);
            table_convergence_fem.set_tex_format("h", "r");

            model.run (cycle);
//             model.output_results (0);
//             exact_fem.output_distributed_solution(model.get_triangulation(), i);
            
            Vector<double> diff_vector;
            std::pair<double,double> l2_norm_dif_fem;
            l2_norm_dif_fem = model.integrate_difference(diff_vector, exact_solution_fem);
            
            table_convergence_fem.add_value("L2",l2_norm_dif_fem.second);
            table_convergence_fem.set_tex_caption("L2","$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$");
            table_convergence_fem.set_precision("L2", 2);
            table_convergence_fem.set_scientific("L2",true);
            
            table_convergence_fem.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
            table_convergence_fem.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
            
            table_convergence_fem.add_value("ndofs",model.get_number_of_dofs());
            table_convergence_fem.add_value("It_{FEM}",model.solver_iterations());
            
            //write the table every cycle (to have at least some results if program fails)
            table_convergence_fem.write_text(std::cout);
            std::ofstream out_file;
            out_file.open(output_dir + model.name() + ".tex");
            table_convergence_fem.write_tex(out_file);
            out_file.close();
            
            out_file.open(output_dir + model.name() + ".txt");
            table_convergence_fem.write_text(out_file, 
                                        TableHandler::TextOutputFormat::table_with_separate_column_description);
            out_file.close();
        }
        return;
    }

  
    XModel xmodel(wells, "",n_aquifers);  
    xmodel.set_name(test_name + "sgfem");
    xmodel.set_enrichment_method(Enrichment_method::sgfem);
//   xmodel.set_name(test_name + "xfem_shift");
//   xmodel.set_enrichment_method(Enrichment_method::xfem_shift);

    xmodel.set_output_dir(output_dir);
    xmodel.set_area(down_left,up_right);
    xmodel.set_transmisivity(transmisivity,0);
    xmodel.set_initial_refinement(initial_refinement);
    xmodel.set_adaptivity(true);
    xmodel.set_enrichment_radius(enrichment_radius);
    xmodel.set_grid_create_type(ModelBase::rect);
    xmodel.set_dirichlet_function(exact_solution);
    xmodel.set_rhs_function(rhs_function);
    xmodel.set_output_options(ModelBase::output_gmsh_mesh
                            | ModelBase::output_solution
                            | ModelBase::output_decomposed
//                             | ModelBase::output_adaptive_plot
                            | ModelBase::output_error
//                             | ModelBase::output_matrix
                            );

    ExactModel exact(exact_solution);
    ConvergenceTable table_convergence;
        
    for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
        table_convergence.add_value("Cycle",cycle);
        table_convergence.set_tex_format("Cycle", "r");
        double h = 2*p_a / pow(2.0,initial_refinement+cycle);
        table_convergence.add_value("h",h);
        table_convergence.set_tex_format("h", "r");
        
        std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
        xmodel.run (cycle);  
//         xmodel.output_results(cycle);
        
        std::cout << "===== XModel_simple finished =====" << std::endl;
      
        std::pair<double,double> l2_norm_dif_xfem;
        Vector<double> diff_vector;
        l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, exact_solution);
        
        table_convergence.add_value("X_L2",l2_norm_dif_xfem.second);
        table_convergence.set_tex_caption("X_L2","$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$");
        table_convergence.set_precision("X_L2", 2);
        table_convergence.set_scientific("X_L2",true);
        
        table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate);
        table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate_log2);
      
        table_convergence.add_value("XFEM-dofs",
                                    xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
        table_convergence.add_value("XFEM-enriched dofs",xmodel.get_number_of_dofs().second);
        table_convergence.add_value("It_{XFEM}",xmodel.solver_iterations());
        
        table_convergence.add_value("XFEM-time",xmodel.last_run_time());
        table_convergence.set_precision("XFEM-time", 3);
        
        table_convergence.set_tex_format("XFEM-dofs", "r");
        table_convergence.set_tex_format("XFEM-enriched dofs", "r");
        table_convergence.set_tex_format("It_{XFEM}", "r");
        table_convergence.set_tex_format("XFEM-time", "r");
      
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

//         xmodel.compute_interpolated_exact(exact_solution);
//         xmodel.output_results(cycle);
//         exact.output_distributed_solution(xmodel.get_output_triangulation(), cycle);
//         xmodel.output_distributed_solution(model.get_triangulation(),cycle, 1);
        exact.output_distributed_solution(xmodel.get_triangulation(), cycle);
    }
    
    delete rhs_function;
    delete exact_solution;
  //*/
  std::cout << "\n\n:::::::::::::::: TWO WELLS TEST END ::::::::::::::::\n\n" << std::endl;
  
}


void test_five_wells(std::string output_dir)
{
  std::cout << "\n\n:::::::::::::::: five WELLS TEST ::::::::::::::::\n\n" << std::endl;
  bool fem = false;
  //------------------------------SETTING----------------------------------
//   std::string test_name = "two_wells_enr_";
  std::string test_name = "five_wells_";
  
  double p_a = 10.0,    //area of the model
         well_radius = 0.003,
//          enrichment_radius = 2.0,
         enrichment_radius = 0.8,
         k_wave_num = 1,
         amplitude = -8;//80;
         
  unsigned int n_aquifers = 1, 
               initial_refinement = 3,
               n_well_q_points = 200;
  
//   double transmisivity = 1;
  double transmisivity = 1e-3;
  
//   std::vector<double> well_parameters ({-179.2814262741238, -168.6628868270077, -208.9034288055592, -184.6703966854124, -206.7745243085635}); 
  std::vector<double> well_parameters ({24.70742701367468, 9.51933747930878, -13.05251754572122, 11.26846892149049, -10.87966163269893}); 
  
  
  //--------------------------END SETTING----------------------------------
  
    Point<2> down_left(0,0);
    Point<2> up_right(p_a, p_a);
    std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
        
    //vector of wells
    std::vector<Well*> wells;
    
    wells.push_back( new Well( well_radius,
                                Point<2>(2.8,2.5)));
    wells.push_back( new Well( well_radius,
                                Point<2>(4.9,5.4)));
    wells.push_back( new Well( well_radius,
                                Point<2>(2.9,7.4)));
    wells.push_back( new Well( well_radius,
                                Point<2>(7.3,7.8)));
    wells.push_back( new Well( well_radius,
                                Point<2>(7.4,2.8)));
    
    wells[0]->set_pressure(-150);
    wells[1]->set_pressure(-30);
    wells[2]->set_pressure(120);
    wells[3]->set_pressure(-50);
    wells[4]->set_pressure(100);
    
    wells[0]->set_perm2aquifer(0, 20);
    wells[1]->set_perm2aquifer(0, 10);
    wells[2]->set_perm2aquifer(0, 10);
    wells[3]->set_perm2aquifer(0, 10);
    wells[4]->set_perm2aquifer(0, 20);
  
//   wells[0]->set_perm2aquitard({1e7, 0});
//   wells[1]->set_perm2aquitard({1e7, 0});  
//   wells[2]->set_perm2aquitard({1e7, 0});  
//   wells[3]->set_perm2aquitard({1e7, 0});  
//   wells[4]->set_perm2aquitard({1e7, 0});
  
    for(unsigned int w=0; w < wells.size(); w++)
    {
        wells[w]->set_perm2aquitard({1e7, 0});
        wells[w]->evaluate_q_points(n_well_q_points);
    }
  
    compare::ExactSolutionMultiple *exact_solution = new compare::ExactSolutionMultiple(k_wave_num, amplitude);
    exact_solution->set_wells(wells, well_parameters, well_parameters);
    Function<2> *rhs_function = new compare::SourceMultiple(transmisivity, *exact_solution);
    unsigned int n_cycles = 10;

    //FEM model creation
    if(fem) //computing H1 norm of regular part of solution
    {
        std::vector<Well*> fem_wells;
        
        for(unsigned int w=0; w < wells.size(); w++)
        {
            fem_wells.push_back(new Well(wells[w]));
//             fem_wells[w]->set_inactive();
            fem_wells[w]->evaluate_q_points(n_well_q_points);
        }

        compare::ExactSolutionMultiple *exact_solution_fem = new compare::ExactSolutionMultiple(k_wave_num, amplitude);
        exact_solution_fem->set_wells(fem_wells, well_parameters, well_parameters);
        Function<2> *rhs_function_fem = new compare::SourceMultiple(transmisivity, *exact_solution_fem);
    
        Model model(fem_wells);    
//     model.set_name(test_name + "fem");
        model.set_name(test_name + "fem_reg");
        model.set_output_dir(output_dir);
        model.set_area(down_left,up_right);
        model.set_transmisivity(transmisivity,0);
        model.set_initial_refinement(initial_refinement);  
//         model.set_ref_coarse_percentage(1.0,0.0);
        model.set_ref_coarse_percentage(0.4,0.1);
        model.set_grid_create_type(ModelBase::rect);
        model.set_dirichlet_function(exact_solution_fem);
        model.set_rhs_function(rhs_function_fem);
        model.set_adaptivity(true);
        model.set_output_options(ModelBase::output_gmsh_mesh
                                | ModelBase::output_solution
                                | ModelBase::output_error);
    
//         model.run (0);
//         model.output_results (0);
        ExactModel exact_fem(exact_solution_fem);
        
        ConvergenceTable table_convergence_fem;
        
        for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
        { 
            table_convergence_fem.add_value("Cycle",cycle);
            table_convergence_fem.set_tex_format("Cycle", "r");
            double h = 2*p_a / pow(2.0,initial_refinement+cycle);
            table_convergence_fem.add_value("h",h);
            table_convergence_fem.set_tex_format("h", "r");

            model.run(cycle);
//             model.output_results(cycle);
            exact_fem.output_distributed_solution(model.get_triangulation(), cycle);
            
            Vector<double> diff_vector;
            std::pair<double,double> l2_norm_dif_fem;
            l2_norm_dif_fem = model.integrate_difference(diff_vector, exact_solution_fem);
            
            table_convergence_fem.add_value("L2",l2_norm_dif_fem.second);
            table_convergence_fem.set_tex_caption("L2","$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$");
            table_convergence_fem.set_precision("L2", 2);
            table_convergence_fem.set_scientific("L2",true);
            
            table_convergence_fem.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
            table_convergence_fem.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
            
            table_convergence_fem.add_value("ndofs",model.get_number_of_dofs());
            table_convergence_fem.add_value("It_{FEM}",model.solver_iterations());
            
            //write the table every cycle (to have at least some results if program fails)
            table_convergence_fem.write_text(std::cout);
            std::ofstream out_file;
            out_file.open(output_dir + model.name() + ".tex");
            table_convergence_fem.write_tex(out_file);
            out_file.close();
            
            out_file.open(output_dir + model.name() + ".txt");
            table_convergence_fem.write_text(out_file, 
                                        TableHandler::TextOutputFormat::table_with_separate_column_description);
            out_file.close();
        }
        return;
    }

  
    XModel xmodel(wells, "",n_aquifers);  
    xmodel.set_name(test_name + "sgfem");
    xmodel.set_enrichment_method(Enrichment_method::sgfem);
//     xmodel.set_name(test_name + "xfem_shift");
//     xmodel.set_enrichment_method(Enrichment_method::xfem_shift);

    xmodel.set_output_dir(output_dir);
    xmodel.set_area(down_left,up_right);
    xmodel.set_transmisivity(transmisivity,0);
    xmodel.set_initial_refinement(initial_refinement);
    xmodel.set_adaptivity(true);
    xmodel.set_enrichment_radius(enrichment_radius);
    xmodel.set_grid_create_type(ModelBase::rect);
    xmodel.set_dirichlet_function(exact_solution);
    xmodel.set_rhs_function(rhs_function);
    xmodel.set_output_options(ModelBase::output_gmsh_mesh
                            | ModelBase::output_solution
                            | ModelBase::output_decomposed
//                             | ModelBase::output_adaptive_plot
                            | ModelBase::output_error
//                             | ModelBase::output_matrix
                            );

    ExactModel exact(exact_solution);
    ConvergenceTable table_convergence;
        
    for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
        table_convergence.add_value("Cycle",cycle);
        table_convergence.set_tex_format("Cycle", "r");
        double h = 2*p_a / pow(2.0,initial_refinement+cycle);
        table_convergence.add_value("h",h);
        table_convergence.set_tex_format("h", "r");
        
        std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
        xmodel.run (cycle);  
//         xmodel.output_results(cycle);
        
        std::cout << "===== XModel_simple finished =====" << std::endl;
      
        std::pair<double,double> l2_norm_dif_xfem;
        Vector<double> diff_vector;
        l2_norm_dif_xfem = xmodel.integrate_difference(diff_vector, exact_solution);
        
        table_convergence.add_value("X_L2",l2_norm_dif_xfem.second);
        table_convergence.set_tex_caption("X_L2","$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$");
        table_convergence.set_precision("X_L2", 2);
        table_convergence.set_scientific("X_L2",true);
        
        table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate);
        table_convergence.evaluate_convergence_rates("X_L2", ConvergenceTable::reduction_rate_log2);
      
        table_convergence.add_value("XFEM-dofs",
                                    xmodel.get_number_of_dofs().first+xmodel.get_number_of_dofs().second);
        table_convergence.add_value("XFEM-enriched dofs",xmodel.get_number_of_dofs().second);
        table_convergence.add_value("It_{XFEM}",xmodel.solver_iterations());
        
        table_convergence.add_value("XFEM-time",xmodel.last_run_time());
        table_convergence.set_precision("XFEM-time", 3);
        
        table_convergence.set_tex_format("XFEM-dofs", "r");
        table_convergence.set_tex_format("XFEM-enriched dofs", "r");
        table_convergence.set_tex_format("It_{XFEM}", "r");
        table_convergence.set_tex_format("XFEM-time", "r");
      
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

//         xmodel.compute_interpolated_exact(exact_solution);
//         xmodel.output_results(cycle);
//         exact.output_distributed_solution(xmodel.get_output_triangulation(), cycle);
//         xmodel.output_distributed_solution(model.get_triangulation(),cycle, 1);
        exact.output_distributed_solution(xmodel.get_triangulation(), cycle);
    }
    
    delete rhs_function;
    delete exact_solution;
  //*/
  std::cout << "\n\n:::::::::::::::: FIVE WELLS TEST END ::::::::::::::::\n\n" << std::endl;
  
}

int main ()
{
  std::string input_dir = "../input/";
  std::string output_dir = "../output/";
  
  GlobalSettingWriter glob;
  glob.write_global_setting(std::cout);
  
  //bedrichov_tunnel(); 
  //return 0;
  
//   find_problem(output_dir);
//   test_adaptive_integration(output_dir);
//   test_adaptive_integration2(output_dir);
//   test_adaptive_integration3(output_dir);
  //test_squares();
//   test_solution(output_dir);
  //test_circle_grid_creation(input_dir);
//    test_convergence_square(output/*_dir);
//     test_radius_convergence_square(o*/utput_dir);
//     test_radius_convergence_sin(output_dir);
//   test_convergence_sin(output_dir);
//   test_convergence_sin_2(output_dir);
//      test_convergence_sin_3(output_dir);
//   test_convergence_sin_4(output_dir);
//   test_convergence_sin_5(output_dir);
//   test_convergence_5_rhow(output_dir);
//   test_convergence_sin_6(output_dir);
//   test_convergence_sin_7(output_dir);
//   test_square_sin_7_rhow(output_dir);
//   test_multiple_wells(output_dir);
//     test_radius_convergence_05(output_dir);
//   test_two_wells(output_dir);
//   test_five_wells(output_dir);
  
  test_multiple_wells2(output_dir);
//   test_two_aquifers(output_dir);
//   test_output(output_dir);
//    test_enr_error(output_dir);
//   test_wells_in_element(output_dir);
//   test_xquadrature_well(output_dir);
//   test_xquadrature_well_2(output_dir);
//     test_xquadrature_well_band(output_dir);
//   visualize_source_term();
  
//   test_polar_radius_sin_3(output_dir);
  return 0;
}

