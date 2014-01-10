#include "model.hh"
#include "xmodel.hh"
#include "simple_models.hh"
#include "bem_model.hh"
#include "comparing.hh"
#include "well.hh"
#include "parameters.hh"
#include "exact_model.hh"
#include <deal.II/base/table_handler.h>

using namespace std;



#include <deal.II/grid/grid_in.h> 
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/persistent_tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include "system.hh"

#include <fstream>
#include <iostream>

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
  tunnel_adapt_linear.set_refinement(1);  
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
  tunnel_xfem.set_refinement(4);                                     
  tunnel_xfem.set_enrichment_radius(50);
  tunnel_xfem.set_output_features();
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


int main ()
{
  std::string input_dir = "../input/";
  std::string output_dir = "../output/";
  //bedrichov_tunnel(); 
  //return 0;
  
  test_circle_grid_creation(input_dir);
  
  /** *******************************
    * Preparation of the models.
    * *******************************/
  
  //which test are to be run
  bool model_simple_mesh_create = 0,
       model_simple_mesh_circle_create = 0,
       
       test_transmisivity = 0,
       test_transmisivity_circle = 0,
       test_perm2fer = 0,
       test_perm2fer_circle = 0,
       
       test_enrichment = 0,
       test_enrichment_circle = 0,
       test_convergence = 1,
       test_permeability = 0,
       
       test_two_enrichment = 0,
       test_two_refinement = 0,
       test_five_wells = 0,
       
       bem_model_create = 0;
  
  //number of aquifers
  unsigned int n_aquifers = 1;
  
  //area of the model
  Point<2> down_left((-2)*Parameters::sqr,(-2)*Parameters::sqr);
  Point<2> up_right(2*Parameters::sqr,2*Parameters::sqr);
  std::cout << "area of the model: " << down_left << "\t" << up_right << std::endl;
  
  Point<2> center(0,0);
  double radius = 10;
  
  Function<2> *zero_dirichlet_bc = new ZeroFunction<2>();
  
  std::string coarse_file_1 = input_dir + "coarse_grid.msh";
  std::string ref_flags_file_1 = input_dir + "ref_flags_12.ptf";
  std::string ref_flags_file_circle_1 = input_dir + "circle_grid_flags_1.ptf";
  std::string convergence_flags_file = input_dir + "convergence_flags_10.ptf";  //9,10,11
  
  
  
  Triangulation<2> coarse_tria_1;
  PersistentTriangulation<2> *tria_1;
  
  std::ifstream in;
  GridIn<2> gridin;
  in.open(coarse_file_1.c_str());
  gridin.attach_triangulation(coarse_tria_1);
  if(in.is_open())
  {
    //reading data from filestream
    gridin.read_msh(in);
  }          
  else
  {
    xprintf(Err, "Could not open coarse grid file: %s", coarse_file_1.c_str());
  }
        
  tria_1 = new PersistentTriangulation<2>(coarse_tria_1);
  in.close();
  in.clear();
  in.open(ref_flags_file_1.c_str());
  if(in.is_open())
    tria_1->read_flags(in);
  else
  {
    xprintf(Warn, "Could not open refinement flags file: %s\n Ingore this if loading mesh without refinement flag file.", ref_flags_file_1.c_str());
  }
  tria_1->restore();
  
  std::cout << "grid 1 loaded" << std::endl;
  //*/
  /** *******************************
    * Temporary model BEM
    * *******************************/
  
  /*
  Bem_model bem_model(wells,n_aquifers);
  
  bem_model.set_area(down_left,up_right);
  bem_model.set_name("bem_model");
  bem_model.set_output_dir("output");
  bem_model.set_transmisivity(Parameters::transmisivity,0);
  bem_model.set_bem_mesh("../bem_meshes/bem_mesh.msh");
  
  bem_model.run();
  
  bem_model.output_distributed_solution("../model_meshes/grid_7.msh");
  
  //*/
  
  /** *******************************
    * Simple one well model with Dirichlet
    * *******************************/

  Well *well = new Well( Parameters::radius,
                         Point<2>(0,0),
                         //Point<2>(-0.2,-0.2),
                         Parameters::perm2fer, 
                         Parameters::perm2tard);
  well->set_pressure(Parameters::pressure_at_top);
  
  //well->evaluate_q_points(Parameters::n_q_points);
  well->evaluate_q_points(500);
  
  //EXACT solution on circle
  ExactModel exact_model(well,radius);
  
  //FEM model creation
  Model_simple model_simple(well);  
  model_simple.set_area(down_left,up_right);
  model_simple.set_name("adapt_model_simple");
  model_simple.set_output_dir(output_dir);
  model_simple.set_transmisivity(Parameters::transmisivity,0);
  model_simple.set_refinement(1);  
  model_simple.set_ref_coarse_percentage(0.3,0.05);
  model_simple.set_dirichlet_function(zero_dirichlet_bc);
  
  XModel_simple xmodel_simple(well);  
  xmodel_simple.set_area(down_left,up_right);
  xmodel_simple.set_name("xmodel_simple");
  xmodel_simple.set_output_dir(output_dir);
  xmodel_simple.set_transmisivity(Parameters::transmisivity,0);
  xmodel_simple.set_refinement(3);                                     
  xmodel_simple.set_enrichment_radius(2.0);
  xmodel_simple.set_output_features();
  xmodel_simple.set_dirichlet_function(zero_dirichlet_bc);
  
  
  ///TEST of enrichment radius
  if(test_enrichment)
  {
  std::cout << "\n\n:::::::::::::::: ENRICHMENT TEST ::::::::::::::::" << std::endl;
  //reference fem solution
  model_simple.set_name("xmodel_simple_enrichment");
  model_simple.set_grid_create_type(Model_base::load);
  model_simple.set_adaptivity(false); 
  model_simple.set_transmisivity(1e-4,0);
  model_simple.set_computational_mesh(coarse_file_1, ref_flags_file_1);
  model_simple.run();
  model_simple.output_results();
  //end of reference fem solution
  
  xmodel_simple.set_name("xmodel_simple_enrichment");
  xmodel_simple.set_transmisivity(1e-4,0);
  xmodel_simple.set_refinement(5);
  xmodel_simple.set_adaptivity(false);
  
  TableHandler table;
  std::vector<double> enr_radius = {0.5, 1.5, 2.0, 3.0, 4.0};
  for(unsigned int i=0; i< enr_radius.size(); i++)
  {
    std::cout << "\n===== XModel_simple running   " << i << "   =====" << std::endl;
    xmodel_simple.set_enrichment_radius(enr_radius[i]);
    xmodel_simple.run (i);     
    xmodel_simple.output_distributed_solution(model_simple.get_triangulation(),i);
    table.add_value("i",i);
    table.add_value("Enrichment radius",enr_radius[i]);
    table.add_value("Error",Comparing::L2_norm_diff(model_simple.get_solution(), 
                                                    xmodel_simple.get_distributed_solution(), 
                                                    model_simple.get_triangulation())
                            / Comparing::L2_norm(model_simple.get_solution(), model_simple.get_triangulation()));
    
    table.write_text(std::cout);
  }
  
  table.set_precision("Enrichment radius",1);
  table.set_precision("Error",3);
  table.set_scientific("Error",true);
    
  std::ofstream out_file;
    
  out_file.open(output_dir + "table_enrichment.tex");
      
  table.write_tex(out_file);
  out_file.close();
  
  std::cout << "\n\n:::::::::::::::: ENRICHMENT TEST - DONE ::::::::::::::::\n\n" << std::endl;
  }
  
  
  ///TEST of enrichment radius on circle mesh
  if(test_enrichment_circle)
  {
  std::cout << "\n\n:::::::::::::::: ENRICHMENT TEST CIRCLE ::::::::::::::::" << std::endl;
  
  xmodel_simple.set_name("xmodel_simple_enrichment_circle");
  xmodel_simple.set_transmisivity(1e-4,0);
  xmodel_simple.set_refinement(3);
  xmodel_simple.set_adaptivity(false);
  
  std::string flag_file = input_dir + "circle_enrichment_flags.ptf";
  xmodel_simple.set_grid_create_type(Model_base::load_circle);
  xmodel_simple.set_computational_mesh_circle(flag_file, center, radius);
  
  std::cout << "computing exact solution on fine mesh..." << std::endl;
  exact_model.output_distributed_solution(convergence_flags_file);
  double exact_norm = Comparing::L2_norm_exact(exact_model.get_triangulation(),well,radius);
  
  std::cout << "L2_norm of the exact solution: " << 
      exact_norm << std::endl;
      
  TableHandler table;
  std::vector<double> enr_radius = {0.2, 0.5, 1.0, 1.5, 2.0, 2.5};
  for(unsigned int i=0; i< enr_radius.size(); i++)
  {
    std::cout << "\n===== XModel_simple running   " << i << "   =====" << std::endl;
    xmodel_simple.set_enrichment_radius(enr_radius[i]);
    xmodel_simple.run (i);     
    xmodel_simple.output_distributed_solution(exact_model.get_triangulation(),i);
    table.add_value("i",i);
    table.add_value("Enrichment radius",enr_radius[i]);
    table.add_value("Rel. Error",Comparing::L2_norm_diff( xmodel_simple.get_distributed_solution(),
                                                          exact_model.get_triangulation(),
                                                          well,
                                                          radius) 
                                 / exact_norm);
    
    table.set_precision("Enrichment radius",1);
    table.set_precision("Rel. Error",3);
    table.set_scientific("Rel. Error",true);
    table.write_text(std::cout);
    
    std::ofstream out_file;
    
    out_file.open(output_dir + "table_enrichment_circle.tex");
      
    table.write_tex(out_file);
    out_file.close();
  }
  
  std::cout << "\n\n:::::::::::::::: ENRICHMENT TEST CIRCLE - DONE ::::::::::::::::\n\n" << std::endl;
  }
  
  
  ///wrong enrichment test
  /*
  xmodel_simple.set_name("xmodel_simple_debug");
  xmodel_simple.set_transmisivity(1e-4,0);
  xmodel_simple.set_refinement(4);
  xmodel_simple.set_enrichment_radius(0.1);
  //xmodel_simple.set_grid_create_type(Model_base::load_circle);
  //xmodel_simple.set_computational_mesh_circle(ref_flags_file_circle_1, center, radius);
  xmodel_simple.run ();     
  
  xmodel_simple.output_distributed_solution(coarse_file_1,ref_flags_file_1);
  //*/
  
  
  /*
  xmodel_simple.set_name("xmodel_simple_debug");
  xmodel_simple.set_transmisivity(1e-4,0);
  xmodel_simple.set_refinement(4);
  xmodel_simple.set_enrichment_radius(2.0);
  //xmodel_simple.set_grid_create_type(Model_base::load_circle);
  //xmodel_simple.set_computational_mesh_circle(ref_flags_file_circle_1, center, radius);
  xmodel_simple.run ();     
  
  xmodel_simple.output_distributed_solution(coarse_file_1,ref_flags_file_1);
  //*/
  /*
  exact_model.output_distributed_solution(convergence_flags_file);
  xmodel_simple.output_distributed_solution(exact_model.get_triangulation());  
  
  xmodel_simple.set_transmisivity(1e-2,0);
  xmodel_simple.run (1);     
  xmodel_simple.output_distributed_solution(exact_model.get_triangulation(),1);
  //*/
  /*
  model_simple.set_name("model_simple_debug");
  model_simple.set_transmisivity(1e-1,0);
  model_simple.set_grid_create_type(Model_base::load_circle);
  model_simple.set_computational_mesh_circle(convergence_flags_file, center, radius);
  
  model_simple.run (0);     
  
  model_simple.output_results(0);
  
  model_simple.set_transmisivity(1e-2,0);
  model_simple.run (1);     
  model_simple.output_results(1);
  */
  
  
  //xmodel_simple.output_distributed_solution(coarse_file_1,ref_flags_file_1);  
  
  //xmodel_simple.set_computational_mesh_circle(ref_flags_file_circle_1, center, radius);
  /*
  exact_model.output_distributed_solution(convergence_flags_file);
  unsigned int n_cycles = 10;
  double transmisivity = 100;
  for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    {  
      std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      xmodel_simple.set_transmisivity(transmisivity,0);
      std::cout << std::scientific << "transmisivity = " << xmodel_simple.get_transmisivity(0) << std::endl;
      
      xmodel_simple.run (cycle);      
      xmodel_simple.output_distributed_solution(coarse_file_1,ref_flags_file_1, false, cycle);
      
      //xmodel_simple.output_distributed_solution(exact_model.get_triangulation(), cycle);  
      //xmodel_simple.output_results(cycle);
      std::cout << "===== XModel_simple finished =====" << std::endl;
      transmisivity *= 0.1;
    } 
  //*/
  
  if(model_simple_mesh_create)
  {
  ///TEST Creation of very fine grid..sofar up to cycle 13  .............................. FEM only
  model_simple.set_adaptivity(true);
  for (unsigned int cycle=0; cycle < 15; ++cycle)
    { 
      std::cout << "===== Model running   " << cycle << "   =====" << std::endl;
      model_simple.run (cycle);
      model_simple.output_results (cycle);
      std::cout << "===== Model finished =====" << std::endl;
    }
  }
  
  if(model_simple_mesh_circle_create)
  {
  ///TEST Creation of very fine grid..sofar up to cycle 12 .............................. FEM only
  model_simple.set_adaptivity(true);
  model_simple.set_name("adapt_model_simple_circle");
  model_simple.set_ref_coarse_percentage(0.3,0.1);
  model_simple.set_computational_mesh_circle(ref_flags_file_circle_1, center, radius);
  
  TableHandler table_circle;
  
  //std::cout << "computing exact solution on fine mesh..." << std::endl;
  //exact_model.output_distributed_solution(convergence_flags_file);
  
  for (unsigned int cycle=0; cycle < 12; ++cycle)
    { 
      std::cout << "===== Model running   " << cycle << "   =====" << std::endl;
      model_simple.run (cycle);
      model_simple.output_results (cycle);
      //exact_model.output_distributed_solution(model_simple.get_triangulation(),cycle);
      std::cout << "===== Model finished =====" << std::endl;
      
      table_circle.add_value("Cycle",cycle);
      table_circle.add_value("$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$",Comparing::L2_norm_diff(model_simple.get_solution(),
                                                                                                model_simple.get_triangulation(),
                                                                                                well,
                                                                                                radius));
      table_circle.add_value("L2_norm_FEM",model_simple.get_solution().l2_norm());
      //table_circle.add_value("L2_norm_exact",exact_model.get_solution().l2_norm());
      
      
      table_circle.set_precision("$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$", 2);
      table_circle.set_scientific("$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$",true);
      table_circle.set_precision("L2_norm_FEM", 2);
      //table_circle.set_precision("L2_norm_exact", 2);
      //output
      table_circle.write_text(std::cout);
    
      std::ofstream out_file;
    
      if(test_perm2fer)
        out_file.open("output/table_circle.tex");
      if(test_perm2fer_circle)
        out_file.open("output/table_circle_circle.tex");
      
      table_circle.write_tex(out_file);
      out_file.close();
    }
  }
  //*/

  
  if(test_convergence)                             //   .............................. FEM, XFEM
  {
  ///TEST CONVERGENCE
  //Parameters of well:
  //pressure = 2;
  //transmisivity = 1e-5;
  //perm2fer = 1e5;
  //perm2tard = 1e10;
  double transmisivity = 1e-4;
  model_simple.set_grid_create_type(Model_base::load_circle);
  model_simple.set_adaptivity(true); 
  model_simple.set_transmisivity(transmisivity,0);
  model_simple.set_computational_mesh_circle(ref_flags_file_circle_1,center, radius);
  model_simple.set_name("model_simple_convergence");
  
  xmodel_simple.set_name("xmodel_simple_convergence");
  xmodel_simple.set_grid_create_type(Model_base::load_circle);
  xmodel_simple.set_transmisivity(transmisivity,0);
  xmodel_simple.set_adaptivity(true);
  xmodel_simple.set_computational_mesh_circle(ref_flags_file_circle_1, center, radius);
  
  xmodel_simple.set_enrichment_radius(2.0);
  
  unsigned int n_cycles = 11;
  double l2_norm_dif_xfem, l2_norm_dif_fem;
  
  std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST ::::::::::::::::\n\n" << std::endl;
  TableHandler table_convergence;

  std::cout << "computing exact solution on fine mesh..." << std::endl;
  exact_model.output_distributed_solution(convergence_flags_file);
  
  std::cout << "L2_norm of the exact solution: " << 
      Comparing::L2_norm_exact(exact_model.get_triangulation(),well,radius) << std::endl;
  
  for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
      std::cout << "===== FEM Model_simple running   " << cycle << "   =====" << std::endl;
      model_simple.set_transmisivity(transmisivity,0);
      std::cout << std::scientific << "transmisivity = " << model_simple.get_transmisivity(0) << std::endl;
      
      model_simple.run (cycle);
      model_simple.output_distributed_solution(exact_model.get_triangulation(), cycle);
      //model_simple.output_results (cycle);
      std::cout << "===== FEM Model_simple finished =====" << std::endl;
      
      //if(cycle < 5)
      {
      std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
      xmodel_simple.run (cycle);  
      xmodel_simple.output_distributed_solution(exact_model.get_triangulation(),cycle);
      std::cout << "===== XModel_simple finished =====" << std::endl;
      
      
      
      l2_norm_dif_xfem = Comparing::L2_norm_diff( xmodel_simple.get_distributed_solution(),
                                                  exact_model.get_triangulation(),
                                                  well,
                                                  radius);
      
      }
      
      l2_norm_dif_fem = Comparing::L2_norm_diff( model_simple.get_distributed_solution(),
                                                 exact_model.get_triangulation(),
                                                 well,
                                                 radius);
      
      table_convergence.add_value("Cycle",cycle);
      table_convergence.add_value("$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$",l2_norm_dif_xfem);
      table_convergence.add_value("$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$",l2_norm_dif_fem);
      
      
      //table_convergence.add_value("XFEM-cells",xmodel_simple.get_triangulation().n_active_cells());
      //table_convergence.add_value("FEM-cells",model_simple.get_triangulation().n_active_cells());
      
      table_convergence.add_value("XFEM-dofs",xmodel_simple.get_number_of_dofs());
      table_convergence.add_value("FEM-dofs",model_simple.get_number_of_dofs());
      
      table_convergence.add_value("XFEM-time",xmodel_simple.get_last_run_time());
      table_convergence.add_value("FEM-time",model_simple.get_last_run_time());
      
      //table_convergence.add_value("L2_norm_exact",exact_model.get_solution().l2_norm());
      //table_convergence.add_value("L2_norm_FEM",model_simple.get_solution().l2_norm());
      //table_convergence.add_value("L2_norm_XFEM",xmodel_simple.get_distributed_solution().l2_norm());
      //table_convergence.add_value("$\\max(x_{FEM})$",model_simple.get_solution().linfty_norm());
      //table_convergence.add_value("$\\max(x_{XFEM})$",xmodel_simple.get_solution().linfty_norm());
      
      
      //write the table every cycle (to have at least some results if program fails)

      table_convergence.set_precision("$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$", 2);
      table_convergence.set_precision("$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$", 2);
      table_convergence.set_scientific("$\\|x_{XFEM}-x_{exact}\\|_{L^2(\\Omega)}$",true);
      table_convergence.set_scientific("$\\|x_{FEM}-x_{exact}\\|_{L^2(\\Omega)}$",true);
      table_convergence.set_precision("XFEM-time", 3);
      table_convergence.set_precision("FEM-time", 3);
      
      //table_convergence.set_precision("L2_norm_exact", 2);
      //table_convergence.set_precision("L2_norm_FEM", 2);
      //table_convergence.set_precision("L2_norm_XFEM", 2);
      
      
      //output
      table_convergence.write_text(std::cout);
      
      std::ofstream out_file;
      out_file.open(output_dir + "table_convergence_circle.tex");
      table_convergence.write_tex(out_file);
      out_file.close();
    } 

    std::cout << "\n\n:::::::::::::::: CONVERGENCE TEST - DONE ::::::::::::::::\n\n" << std::endl;
  } 
  
  
  if(test_permeability)                             //   .............................. FEM, XFEM
  {
  ///TEST PERMEABILITY
  //Parameters of well:
  //pressure = 2;
  //transmisivity = 1e-5;
  //perm2fer = 1e5;
  //perm2tard = 1e10;
  double transmisivity = 1e-4;
  double perm2fer = 1e-4;
  model_simple.set_grid_create_type(Model_base::load);
  model_simple.set_adaptivity(false); 
  model_simple.set_transmisivity(transmisivity,0);
  model_simple.set_computational_mesh(coarse_file_1, ref_flags_file_1);
  model_simple.set_name("model_simple_permeability");
  
  xmodel_simple.set_name("xmodel_simple_permeability");
  xmodel_simple.set_transmisivity(transmisivity,0);
  xmodel_simple.set_adaptivity(false);
  xmodel_simple.set_refinement(4);
  xmodel_simple.set_grid_create_type(Model_base::rect);
  
  xmodel_simple.set_enrichment_radius(2.0);
  
  unsigned int n_cycles = 6;
  
  std::cout << "\n\n:::::::::::::::: PERMEABILITY TEST ::::::::::::::::\n\n" << std::endl;
  TableHandler table;
  
  for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
      well->set_perm2aquifer(perm2fer);
      std::cout << std::scientific << "permeability coeficient well-aquifer sigma = " << well->perm2aquifer() << std::endl;
      
      std::cout << "===== FEM Model_simple running   " << cycle << "   =====" << std::endl;
      model_simple.run (cycle);
      model_simple.output_results(cycle);
      //model_simple.output_distributed_solution(*tria_1,cycle);
      //model_simple.output_results (cycle);
      std::cout << "===== FEM Model_simple finished =====" << std::endl;
      

      std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
      xmodel_simple.run (cycle);  
      xmodel_simple.output_distributed_solution(*tria_1,cycle);
      std::cout << "===== XModel_simple finished =====" << std::endl;
      
      //table.add_value("Cycle",cycle);
      table.add_value("$\\sigma$",perm2fer);
      table.add_value("FEM_max",model_simple.get_solution().linfty_norm());
      table.add_value("XFEM_max",xmodel_simple.get_distributed_solution().linfty_norm());
      
      table.set_precision("$\\sigma$",0);
      table.set_scientific("$\\sigma$",true);
      table.set_precision("FEM_max",3);
      table.set_precision("XFEM_max",3);
      
      perm2fer *= 1e2;
      
      //output
      table.write_text(std::cout);
    
      std::ofstream out_file;
      out_file.open(output_dir + "table_permeability.tex");
      table.write_tex(out_file);
      out_file.close();
      
    } 

    std::cout << "\n\n:::::::::::::::: PERMEABILITY TEST - DONE ::::::::::::::::\n\n" << std::endl;
  } 
  
  
  if(test_transmisivity)                             //   .............................. FEM, XFEM
  {
  ///TEST transmisivity
  //Parameters of well:
  //pressure = 2;
  //transmisivity = 1e-5;
  //perm2fer = 1e5;
  //perm2tard = 1e10;
  model_simple.set_grid_create_type(Model_base::load);
  model_simple.set_adaptivity(false); 
  model_simple.set_computational_mesh(coarse_file_1, ref_flags_file_1);
  model_simple.set_name("model_simple_permeability");
  
  xmodel_simple.set_name("xmodel_simple_permeability");
  xmodel_simple.set_adaptivity(false);
  xmodel_simple.set_refinement(4);
  xmodel_simple.set_grid_create_type(Model_base::rect);
  
  xmodel_simple.set_enrichment_radius(2.0);
  
  double transmisivity = 1e-10;
  double perm2fer = 1e5;
  well->set_perm2aquifer(perm2fer);
  
  unsigned int n_cycles = 7;
  
  Vector<double> xsolution_trans, 
                 solution_trans;
  
  std::cout << "\n\n:::::::::::::::: TRANSMISIVITY TEST ::::::::::::::::\n\n" << std::endl;
  TableHandler table;
  
  for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
      std::cout << std::scientific << "transmisivity = " << transmisivity << std::endl;
      model_simple.set_transmisivity(transmisivity,0);
      xmodel_simple.set_transmisivity(transmisivity,0);
      
      std::cout << "===== FEM Model_simple running   " << cycle << "   =====" << std::endl;
      model_simple.run (cycle);
      model_simple.output_results(cycle);
      //model_simple.output_distributed_solution(*tria_1,cycle);
      //model_simple.output_results (cycle);
      std::cout << "===== FEM Model_simple finished =====" << std::endl;
      

      std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      
      xmodel_simple.run (cycle);  
      xmodel_simple.output_distributed_solution(*tria_1,cycle);
      std::cout << "===== XModel_simple finished =====" << std::endl;
      
      if(cycle == 0)
      {
        xsolution_trans = xmodel_simple.get_distributed_solution();
        solution_trans = model_simple.get_distributed_solution();
      }
      
      //table.add_value("Cycle",cycle);
      table.add_value("$T$",transmisivity);
      table.add_value("Difference FEM",Comparing::L2_norm_diff(solution_trans, 
                                                               model_simple.get_distributed_solution(), 
                                                               *tria_1));
      table.add_value("Difference XFEM",Comparing::L2_norm_diff(xsolution_trans, 
                                                               xmodel_simple.get_distributed_solution(), 
                                                               *tria_1));
 
      table.set_precision("$T$",0);
      table.set_scientific("$T$",true);
      
      table.set_precision("$Difference FEM$",3);
      table.set_scientific("$Difference FEM$",true);
      
      table.set_precision("$Difference XFEM$",3);
      table.set_scientific("$Difference XFEM$",true);
      
      transmisivity *= 1e2;
      
      //output
      table.write_text(std::cout);
    
      std::ofstream out_file;
      out_file.open(output_dir + "table_transmisivity.tex");
      table.write_tex(out_file);
      out_file.close();
      
    } 

    std::cout << "\n\n:::::::::::::::: TRANSMISIVITY TEST - DONE ::::::::::::::::\n\n" << std::endl;
  } 
  
  
  
  

  
  
  
  /*
  if(test_perm2fer || test_perm2fer_circle)
  {
  //TEST PERMEABILITY
  //Parameters of well:
  //pressure = 2;
  double perm2fer = 1e-5;
  model_simple.set_transmisivity(1.0,0);
  model_simple.set_grid_create_type(Model_base::load);
  model_simple.set_adaptivity(false); 
  
  xmodel_simple.set_transmisivity(1.0,0);
  
  if(test_perm2fer)
  {
    model_simple.set_computational_mesh(coarse_file_1, ref_flags_file_1);
    model_simple.set_name("model_simple_perm2fer");
    xmodel_simple.set_name("xmodel_simple_perm2fer");
  }
  if(test_perm2fer_circle)
  {
    model_simple.set_computational_mesh_circle(ref_flags_file_circle_1,center, radius);
    model_simple.set_name("model_simple_perm2fer_circle");
    xmodel_simple.set_name("xmodel_simple_perm2fer_circle");
  }
  
  
  unsigned int n_cycles = 20;
  std::vector<double> l2_norm_difs(n_cycles);
  
  
  std::cout << "\n\n:::::::::::::::: PERMEABILITY TEST 1 ::::::::::::::::\n\n" << std::endl;
  TableHandler table_permeability_1;
  
  for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
      well->set_perm2aquifer(perm2fer);
      std::cout << std::scientific << "permeability coeficient well-aquifer sigma = " << well->get_perm2aquifer() << std::endl;
      
      std::cout << "===== FEM Model_simple running   " << cycle << "   =====" << std::endl;
      model_simple.run (cycle);
      model_simple.output_results (cycle);
      std::cout << "===== FEM Model_simple finished =====" << std::endl;
      
      std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl; 
      xmodel_simple.run (cycle);      
      xmodel_simple.output_distributed_solution(model_simple.get_triangulation(), cycle);
      std::cout << "===== XModel_simple finished =====" << std::endl;
      
      l2_norm_difs.push_back(Comparing::L2_norm_diff( model_simple.get_solution(), 
                                                      xmodel_simple.get_distributed_solution(),
                                                      model_simple.get_triangulation() ) );
      
      table_permeability_1.add_value("Cycle",cycle);
      table_permeability_1.add_value("Permeability",perm2fer);
      table_permeability_1.add_value("$\\|x_{FEM}-x_{XFEM}\\|_{L^2(\\Omega)}$",l2_norm_difs.back());
      table_permeability_1.add_value("L2_norm_FEM",model_simple.get_solution().l2_norm());
      table_permeability_1.add_value("L2_norm_XFEM",xmodel_simple.get_distributed_solution().l2_norm());
      table_permeability_1.add_value("$\\max(x_{FEM})$",model_simple.get_solution().linfty_norm());
      table_permeability_1.add_value("$\\max(x_{XFEM})$",xmodel_simple.get_solution().linfty_norm());
      
      std::cout << "L2_norm of difference between solutions: " 
                << l2_norm_difs.back()
                << std::endl;
      
      perm2fer *= 10;
      
      
      //write the table every cycle (to have at least some results if program fails)
      //table.set_tex_format("numbers", "r");
      table_permeability_1.set_precision("Permeability", 2);
      table_permeability_1.set_scientific("Permeability",true);
      table_permeability_1.set_precision("L2_norm_FEM", 2);
      table_permeability_1.set_precision("$\\|x_{FEM}-x_{XFEM}\\|_{L^2(\\Omega)}$", 2);
      table_permeability_1.set_precision("L2_norm_XFEM", 2);
      table_permeability_1.set_scientific("$\\|x_{FEM}-x_{XFEM}\\|_{L^2(\\Omega)}$",true);
      //output
      table_permeability_1.write_text(std::cout);
      std::ofstream out_file;
  
      if(test_perm2fer)
        out_file.open("output/table_permeability_1.tex");
      if(test_perm2fer_circle)
        out_file.open("output/table_permeability_1_circle.tex");
  
      table_permeability_1.write_tex(out_file);
      out_file.close();
    } 
   
  
  std::cout << "\n\n:::::::::::::::: PERMEABILITY TEST 1 - DONE ::::::::::::::::\n\n" << std::endl;
    
  }
  
  if(test_transmisivity || test_transmisivity_circle)                             //   .............................. FEM, XFEM
  {
  //TEST TRANSMISIVITY
  //Parameters of well:
  //pressure = 2;
  //transmisivity = changing
  //perm2fer = 1e5;
  //perm2tard = 1e10;
  double transmisivity = 100;
  model_simple.set_grid_create_type(Model_base::load);
  model_simple.set_adaptivity(false); 
  
  if(test_transmisivity)
  {
    model_simple.set_computational_mesh(coarse_file_1, ref_flags_file_1);
    model_simple.set_name("model_simple_transmisivity");
    xmodel_simple.set_name("xmodel_simple_transmisivity");
  }
  if(test_transmisivity_circle)
  {
    model_simple.set_computational_mesh_circle(ref_flags_file_circle_1,center, radius);
    model_simple.set_name("model_simple_transmisivity_circle");
    xmodel_simple.set_name("xmodel_simple_transmisivity_circle");
    xmodel_simple.set_grid_create_type(Model_base::circle);
    xmodel_simple.set_refinement(4); 
  }
  
  unsigned int n_cycles = 20;
  std::vector<double> l2_norm_difs(n_cycles);
  
  std::cout << "\n\n:::::::::::::::: TRANSMISIVITY TEST 1 ::::::::::::::::\n\n" << std::endl;
  TableHandler table_transmisivity_1;

  for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    { 
      std::cout << "===== FEM Model_simple running   " << cycle << "   =====" << std::endl;
      model_simple.set_transmisivity(transmisivity,0);
      std::cout << std::scientific << "transmisivity = " << model_simple.get_transmisivity(0) << std::endl;
      
      model_simple.run (cycle);
      model_simple.output_results (cycle);
      std::cout << "===== FEM Model_simple finished =====" << std::endl;
      
      std::cout << "===== XModel_simple running   " << cycle << "   =====" << std::endl;
      xmodel_simple.set_transmisivity(transmisivity,0);
      std::cout << std::scientific << "transmisivity = " << xmodel_simple.get_transmisivity(0) << std::endl;
      
      xmodel_simple.run (cycle);      
      xmodel_simple.output_distributed_solution(model_simple.get_triangulation(), cycle);
      std::cout << "===== XModel_simple finished =====" << std::endl;
      
      l2_norm_difs.push_back(Comparing::L2_norm_diff( model_simple.get_solution(), 
                                                      xmodel_simple.get_distributed_solution(),
                                                      model_simple.get_triangulation() ) );
      
      table_transmisivity_1.add_value("Cycle",cycle);
      table_transmisivity_1.add_value("Transmisivity",transmisivity);
      table_transmisivity_1.add_value("$\\|x_{FEM}-x_{XFEM}\\|_{L^2(\\Omega)}$",l2_norm_difs.back());
      table_transmisivity_1.add_value("L2_norm_FEM",model_simple.get_solution().l2_norm());
      table_transmisivity_1.add_value("L2_norm_XFEM",xmodel_simple.get_distributed_solution().l2_norm());
      table_transmisivity_1.add_value("$\\max(x_{FEM})$",model_simple.get_solution().linfty_norm());
      table_transmisivity_1.add_value("$\\max(x_{XFEM})$",xmodel_simple.get_solution().linfty_norm());
      
      std::cout << "L2_norm of difference between solutions: " 
                << l2_norm_difs.back()
                << std::endl;
      
      transmisivity *= 0.1;
      
      
    //write the table every cycle (to have at least some results if program fails)
    //table.set_tex_format("numbers", "r");
    //table_transmisivity_1.set_precision("Transmisivity", 2);

    table_transmisivity_1.set_scientific("Transmisivity",true);
    table_transmisivity_1.set_precision("L2_norm_FEM", 2);
    table_transmisivity_1.set_precision("$\\|x_{FEM}-x_{XFEM}\\|_{L^2(\\Omega)}$", 2);
    table_transmisivity_1.set_precision("L2_norm_XFEM", 2);
    table_transmisivity_1.set_scientific("$\\|x_{FEM}-x_{XFEM}\\|_{L^2(\\Omega)}$",true);
    //output
    table_transmisivity_1.write_text(std::cout);
    
    std::ofstream out_file;
    
    if(test_perm2fer)
        out_file.open("output/table_transmisivity_1.tex");
    if(test_perm2fer_circle)
        out_file.open("output/table_transmisivity_1_circle.tex");
      
    table_transmisivity_1.write_tex(out_file);
    out_file.close();
    } 

  
    std::cout << "\n\n:::::::::::::::: TRANSMISIVITY TEST 1 - DONE ::::::::::::::::\n\n" << std::endl;
  } 
  //*/
  
  
  
  
  /** *******************************
   * Two well XFEM model
   ** *******************************/
  down_left = Point<2>(-10.0,-5.0);
  up_right = Point<2>(10.0,5.0);
  
  //vector of wells
  std::vector<Well*> wells;
  
  //left injecting well
  wells.push_back( new Well( Parameters::radius,
                             //Point<2>((-5.0)*Parameters::x_dec,0.0),
                             Point<2>((-5.0),0.0),
                             Parameters::perm2fer, 
                             Parameters::perm2tard));
  
  //right pumping well
  wells.push_back( new Well( Parameters::radius,
                             //Point<2>(5.0*Parameters::x_dec,0.0),
                             Point<2>(5.0,0.0), 
                             Parameters::perm2fer, 
                             Parameters::perm2tard));
  //setting BC - pressure at the top of the wells
  wells[0]->set_pressure(Parameters::pressure_at_top);
  wells[1]->set_pressure((-0.1)*Parameters::pressure_at_top);
  
  for(unsigned int w=0; w < wells.size(); w++)
  {
    wells[w]->evaluate_q_points(500);
  }
  
  
  //std::string model_mesh_file_1 = "../model_meshes/grid_7.msh";
  std::string coarse_file_two_1 = input_dir + "coarse_grid_two.msh";
  std::string ref_flags_file_two_1 = input_dir + "ref_flags_two_9.ptf";
  
  
  Model model(wells, "model", n_aquifers);
  
  model.set_area(down_left,up_right);
  model.set_name("adapt_model");
  model.set_output_dir(output_dir);
  model.set_transmisivity(Parameters::transmisivity,0);
  model.set_refinement(Parameters::start_refinement);
  model.set_ref_coarse_percentage(0.3,0.05);
  model.set_adaptivity(true);
  model.set_computational_mesh(coarse_file_two_1, ref_flags_file_two_1);
  
  //model.run();
  //model.output_results();
  
  /* //making of the grid
  unsigned int n_cycles=15;
  for(unsigned int cycle=0; cycle < n_cycles; cycle++)
  {
      model.run(cycle);
      model.output_results(cycle);
  }
  //*/
  
  XModel xmodel(wells, "xmodel", n_aquifers);

  xmodel.set_area(down_left,up_right);
  xmodel.set_output_dir(output_dir);
  xmodel.set_transmisivity(Parameters::transmisivity,0);
  xmodel.set_refinement(5);
  xmodel.set_enrichment_radius(2.0);

  
  ///TEST of enrichment radius
  if(test_two_enrichment)
  {
  std::cout << "\n\n:::::::::::::::: ENRICHMENT TEST TWO ::::::::::::::::" << std::endl;
  //reference fem solution
  model.set_name("xmodel_two_enrichment");
  model.set_grid_create_type(Model_base::load);
  model.set_adaptivity(false); 
  model.set_transmisivity(1e-4,0);
  model.set_computational_mesh(coarse_file_two_1, ref_flags_file_two_1);
  model.run();
  model.output_results();
  //end of reference fem solution
  
  xmodel.set_name("xmodel_two_enrichment");
  xmodel.set_transmisivity(1e-4,0);
  xmodel.set_refinement(5);
  xmodel.set_adaptivity(false);
  
  TableHandler table;
  std::vector<double> enr_radius = {0.5, 1.0, 1.5, 2.0, 3.0};
  for(unsigned int i=0; i< enr_radius.size(); i++)
  {
    std::cout << "\n===== XModel running   " << i << "   =====" << std::endl;
    xmodel.set_enrichment_radius(enr_radius[i]);
    xmodel.run (i);     
    xmodel.output_distributed_solution(model.get_triangulation(),i);
    table.add_value("i",i);
    table.add_value("Enrichment radius",enr_radius[i]);
    table.add_value("Error",Comparing::L2_norm_diff(model.get_solution(), 
                                                    xmodel.get_distributed_solution(), 
                                                    model.get_triangulation()));
    table.add_value("Time", xmodel.get_last_run_time());
    
    table.write_text(std::cout);
  }
  
  table.set_precision("Enrichment radius",1);
  table.set_precision("Error",3);
  table.set_precision("Time",3);
  //table.set_scientific("Error",true);
    
  std::ofstream out_file;
    
  out_file.open(output_dir + "table_two_enrichment.tex");
      
  table.write_tex(out_file);
  out_file.close();
  
  std::cout << "\n\n:::::::::::::::: ENRICHMENT TEST TWO - DONE ::::::::::::::::\n\n" << std::endl;
  }
  
  
  ///TEST of enrichment radius
  if(test_two_refinement)
  {
  std::cout << "\n\n:::::::::::::::: REFINEMENT TEST TWO ::::::::::::::::" << std::endl;
  //reference fem solution
  model.set_name("xmodel_two_refinement");
  model.set_grid_create_type(Model_base::load);
  model.set_adaptivity(false); 
  model.set_transmisivity(1e-4,0);
  model.set_computational_mesh(coarse_file_two_1, ref_flags_file_two_1);
  model.run();
  model.output_results();
  //end of reference fem solution
  
  xmodel.set_name("xmodel_two_refinement");
  xmodel.set_transmisivity(1e-4,0);
  xmodel.set_enrichment_radius(2.0);
  xmodel.set_adaptivity(true);
  
  TableHandler table;
  double refinement=3;
  xmodel.set_refinement(refinement);
  for(unsigned int i=0; i < 3; i++)
  {
    std::cout << "\n===== XModel running   " << i << "   =====" << std::endl;
    
    xmodel.run (i);     
    xmodel.output_distributed_solution(model.get_triangulation(),i);
    //table.add_value("i",i);
    table.add_value("Refinement", refinement+i);
    table.add_value("Rel. Error",Comparing::L2_norm_diff(model.get_solution(), 
                                                    xmodel.get_distributed_solution(), 
                                                    model.get_triangulation())
                                 / Comparing::L2_norm(model.get_solution(),model.get_triangulation()));
    
    table.add_value("Time", xmodel.get_last_run_time());
    
    table.write_text(std::cout);
  }
  
  //table.set_precision("Refinement",0);
  table.set_precision("Rel. Error",3);
  table.set_precision("Time",3);
  //table.set_scientific("Error",true);
    
  std::ofstream out_file;
    
  out_file.open(output_dir + "table_two_refinement.tex");
      
  table.write_tex(out_file);
  out_file.close();
  
  std::cout << "\n\n:::::::::::::::: REFINEMENT TEST TWO - DONE ::::::::::::::::\n\n" << std::endl;
  }
  
  
  //*/
  if(test_five_wells)
  {
    down_left = Point<2>(-10.0,-10.0);
    up_right = Point<2>(10.0,10.0);
    double transmisivity = 1e-4;
  
    //vector of wells
    std::vector<Well*> wells_2;
  
    wells_2.push_back( new Well( Parameters::radius,
                             Point<2>(-5.0,-5.0),
                             Parameters::perm2fer, 
                             Parameters::perm2tard));
  
    wells_2.push_back( new Well( Parameters::radius,
                             Point<2>(0.0,0.0), 
                             Parameters::perm2fer, 
                             Parameters::perm2tard));
    
    wells_2.push_back( new Well( Parameters::radius,
                             Point<2>(-5.0,5.0), 
                             Parameters::perm2fer, 
                             Parameters::perm2tard));
    
    wells_2.push_back( new Well( Parameters::radius,
                             Point<2>(5.0,5.0), 
                             Parameters::perm2fer, 
                             Parameters::perm2tard));
    
    wells_2.push_back( new Well( Parameters::radius,
                             Point<2>(5.0,-5.0), 
                             Parameters::perm2fer, 
                             Parameters::perm2tard));
    
    //setting BC - pressure at the top of the wells
    wells_2[2]->set_pressure(2*Parameters::pressure_at_top);
    wells_2[3]->set_pressure(Parameters::pressure_at_top);
    wells_2[4]->set_pressure(3*Parameters::pressure_at_top);
  
    for(unsigned int w=0; w < wells_2.size(); w++)
    {
      wells_2[w]->evaluate_q_points(500);
    }
    
    Model model_five(wells_2, "model_five", n_aquifers);
  
    model_five.set_area(down_left,up_right);
    model_five.set_output_dir(output_dir);
    model_five.set_transmisivity(transmisivity,0);
    model_five.set_refinement(3);
    model_five.set_ref_coarse_percentage(0.3,0.05);
    model_five.set_adaptivity(true);
    
  
    //model.run();
    //model.output_results();
    
    //making of the grid
    unsigned int n_cycles=10;
    for(unsigned int cycle=0; cycle < n_cycles; cycle++)
    {
      std::cout << "\n===== Model running   " << cycle << "   =====" << std::endl;
      model_five.run(cycle);
      model_five.output_results(cycle);
      std::cout << "computation time = " << model_five.get_last_run_time() << std::endl;
      std::cout << "\n===== Model finished   " << cycle << "   =====" << std::endl;
    }
    //*/
    
    
    XModel xmodel_five(wells_2, "xmodel_five", n_aquifers);

    xmodel_five.set_area(down_left,up_right);
    xmodel_five.set_output_dir(output_dir);
    xmodel_five.set_transmisivity(transmisivity,0);
    xmodel_five.set_refinement(5);
    xmodel_five.set_enrichment_radius(2.0);
  
    xmodel_five.run();
    xmodel_five.output_distributed_solution(model_five.get_triangulation());
    std::cout << "computation time = " << xmodel_five.get_last_run_time() << std::endl;
    std::cout << "relative error to fem solution = " << 
          Comparing::L2_norm_diff(model_five.get_solution(), 
                                  xmodel_five.get_distributed_solution(), 
                                  model_five.get_triangulation())
                                 / Comparing::L2_norm(model_five.get_solution(),model_five.get_triangulation())
              << std::endl;
    //*/
  }

  
  /** *****************************************************
    * Testing bem model computed on differently refined meshes, 
    * evaluating bem model on a fine model mesh and
    * comparing the norm of difference from the bem model
    * computed on the reference bem mesh and evaluated 
    * on the fine model mesh. 
    *******************************************************/
  /*
  std::string model_mesh_file_1 = "../model_meshes/grid_7.msh";
  std::string bem_ref_mesh_file_1 = "../bem_meshes/bem_mesh_07.msh";
  
  std::vector<Point<2> > quad_points;
  quad_points = Comparing::get_all_quad_points(model_mesh_file_1);
  unsigned int n_quad_points = quad_points.size();
  
  //reference bem_model on finest 1D mesh
  Bem_model bem_ref_1;
  bem_ref_1.run(bem_ref_mesh_file_1);
  
  //evaluating on the finest 2D mesh
  std::cout << "computing reference solution..." << std::endl;
  Vector<double> bem_ref_sol_1(n_quad_points);
  bem_ref_sol_1 = bem_ref_1.get_solution_at_points(quad_points);
  
  const unsigned int num_cycle = 7;
  double l2_norms[num_cycle];
  
  for (unsigned int cycle=0; cycle < num_cycle; ++cycle)
    {
      //creation of bem_model
      Bem_model bem_model;
      std::stringstream bem_mesh_file;
      bem_mesh_file << "../bem_meshes/" << "bem_mesh_0" << cycle << ".msh";
      bem_model.run(bem_mesh_file.str(),cycle);
     
      //evaluating solution on the finest 2D mesh
      Vector<double> bem_solution(n_quad_points);
      bem_solution = bem_model.get_solution_at_points(quad_points);
      
      //evaluating L2norm of the diffence to reference bem model
      l2_norms[cycle] = Comparing::L2_norm_diff(bem_solution,bem_ref_sol_1);
      
      //bem_model.output_2d_results(model_mesh, cycle);
    }
  
  
  std::cout << "..........RESULTS.........." << 
    "\n2D mesh:   " << model_mesh_file_1 << 
    "\n1D reference mesh:   " << bem_ref_mesh_file_1 << std::endl;
    
  for (unsigned int i=0; i < num_cycle; ++i)
  {
    std::cout << "L2norm(bem_model_" << i << "):   " << l2_norms[i] << std::endl;
  }
  //*/
  
  
  
  /** *****************************************************
    * Testing bem model's well boundary values convergence with permeability going to inf
    *******************************************************/
  /*
  
  //mesh for computing the model
  std::string bem_mesh_file_2 = "../bem_meshes/bem_mesh_07.msh";
  //2d mesh for output
  std::string model_mesh_file_2 = "../model_meshes/grid_5.msh";
  //number of points on the circle
  const unsigned int n = 100;
  //centers of the circle wells
  Point<2> center_1(-Parameters::x_dec,0.0);
  Point<2> center_2(Parameters::x_dec,0.0);
  //vector of angles (computing points on the circle)
  std::vector<double> angles(n);
  //points on the circle
  std::vector<Point<2> > well_points_1 = Comparing::generate_circle_points(center_1,Parameters::radius,//+1e-9,
                                                                         n, angles);
  std::vector<Point<2> > well_points_2 = Comparing::generate_circle_points(center_2,Parameters::radius,
                                                                         n, angles);
  
  std::vector<Point<2> > around_well_points_1 = Comparing::generate_circle_points(center_1,Parameters::radius+1e-2,
                                                                         n, angles);
  
  Vector<double> bem_sol_2a(n);
  Vector<double> bem_sol_2b(n);
  Vector<double> bem_sol_2c(n);
  
  double permeability = 1e+6;//1.0;
  
  for (unsigned int k=1; k < 2; k++)
  {
    permeability *= k;
    std::cout << "===============" << k << ". " << "permeability = " 
              << permeability << " ==================" << std::endl;
    Bem_model bem_model(permeability);
    bem_model.run(bem_mesh_file_2);
    
    
    bem_model.write_bem_solution();
    
    //std::cout << center_1 << std::endl;
    //std::cout << center_2 << std::endl;
    //std::cout << Parameters::radius/2 << std::endl;
    
    std::vector<Point<2> > boundary_points(2);
    boundary_points[0] = Point<2>(center_1[0]+Parameters::radius, center_1[1]);
    boundary_points[1] = Point<2>(center_1[0], center_1[1]+Parameters::radius);
    bem_model.get_boundary_solution(boundary_points);
    ///
    
    bem_sol_2a = bem_model.get_boundary_solution(well_points_1);
    bem_sol_2b = bem_model.get_boundary_solution(well_points_2);
    bem_sol_2c = bem_model.get_solution_at_points(around_well_points_1);
    
    
    for(unsigned int i=0; i < n; i++)
    {
      std::cout << angles[i] << ":\t" << std::setw(10) << bem_sol_2a[i] 
                << "\t" << std::setw(10) << bem_sol_2b[i] 
                << "\t" << std::setw(10) << bem_sol_2c[i] << std::endl;
    }
    
    bem_model.output_2d_results(model_mesh_file_2,k);
  }
  
  
  //*/
  
  
  
  
  /*
  //deallog.depth_console (10);
  
  Bem_model bem_model;
  bem_model.run();
  
  Model model;
  
  std::vector<Point<2> > support_points;
  Vector<double> bem_model_solution;
      
  for (unsigned int cycle=0; cycle<Parameters::cycle; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << "==============" << std::endl;
      
      std::cout << "===== Model running =====" << std::endl;
      model.run (cycle);
      std::cout << "===== Model finished =====" << std::endl;
      
      std::cout << "===== Evaluating BEM Model's solution in Model's support points =====" << std::endl;
      
      model.get_support_points(support_points);
  
      //for(unsigned int i = 0; i < support_points.size(); i++)
      //  std::cout << support_points[i] << std::endl; 
  
      bem_model_solution = bem_model.get_solution_at_points(support_points);
  
      
      //for(unsigned int i = 0; i < support_points.size(); i++)
      //  std::cout << bem_model_solution(i) << std::endl; 
      
  
      model.output_foreign_results(cycle, bem_model_solution);
    }

  //*/
  
  
  /** *****************************************************
    * Testing model with permeability going to inf
    *******************************************************/  
  /*
  double permeability = 1e10;//1.0;
  
  Model model3(permeability);
  std::cout << "=============== permeability = " 
              << permeability << " ==================" << std::endl;
             
  for (unsigned int cycle=0; cycle < 15; ++cycle)
  {
    std::cout << "===== Model running   " << cycle << "   =====" << std::endl;
      model3.run (cycle);
    std::cout << "===== Model finished =====" << std::endl;
  }
  //*/
  
  /** *****************************************************
    * FOR MAKING VERY FINE MESH FOR COMPUTING L2NORMS OF BEM SOLUTION 
    *******************************************************/  
  /*
  Model model;
  
  for (unsigned int cycle=0; 12; ++cycle)
    { 
      std::cout << "===== Model running   " << cycle << "   =====" << std::endl;
      model.run (cycle);
      std::cout << "===== Model finished =====" << std::endl;
    }
  //*/
  
  /** *****************************************************
    * FOR MAKING FINE MESH FOR XFEM MODEL ON DOMAIN 10x10
    *******************************************************/  
  
  /*
  Model model(wells,n_aquifers);
  model.set_area(down_left,up_right);
  model.set_name("adapt_model");
  model.set_output_dir("output");
  model.set_transmisivity(Parameters::transmisivity,0);
  model.set_refinement(1);
  
  for (unsigned int cycle=0; 15; ++cycle)
    { 
      std::cout << "===== Model running   " << cycle << "   =====" << std::endl;
      model.run (cycle);
      std::cout << "===== Model finished =====" << std::endl;
    }
  //*/
  
  
  
  
  delete zero_dirichlet_bc;
  
  delete tria_1;
  
  for(unsigned int i=0; i < wells.size(); i++)
  {
    delete wells[i];
  }
  delete well;
  
  return 0;
}

