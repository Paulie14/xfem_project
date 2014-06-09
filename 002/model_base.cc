


#include "model_base.hh"
#include "system.hh"
#include <deal.II/base/function.h>

#include <dirent.h>
#include <sys/stat.h>
#include <time.h>
#include <fstream>
#include <iostream>

#include "well.hh"

Model_base::Model_base()
:   grid_create(Model_base::rect),
    down_left(Point<2>(0.0,0.0)),
    up_right(Point<2>(1.0,1.0)),
    dirichlet_function(nullptr),
    rhs_function(nullptr),
    triangulation_changed(true),
    is_adaptive(false),
    init_refinement(0),
    
    n_aquifers(1),
    
    cycle_(-1),
    last_run_time_(0),
    solver_it(0),
    matrix_output_(false),
    sparsity_pattern_output_(false),
    output_dir("../output/model/"),
    main_output_dir("../output/"),
    name("model_base")
{
    prm_declare();
}

Model_base::Model_base(const std::string& name, 
                       const unsigned int& n_aquifers)
  : grid_create(Model_base::rect),
    down_left(Point<2>(0.0,0.0)),
    up_right(Point<2>(1.0,1.0)),
    dirichlet_function(nullptr),
    rhs_function(nullptr),
    triangulation_changed(true),
    is_adaptive(false),
    init_refinement(0),
    
    n_aquifers(n_aquifers),
    
    cycle_(-1),
    last_run_time_(0),
    solver_it(0),
    matrix_output_(false),
    sparsity_pattern_output_(false),
    output_dir("../output/model/"),
    main_output_dir("../output/"),
    name(name)
    
{
  transmisivity.resize(n_aquifers, 1.0);
  prm_declare();
}

Model_base::Model_base(const std::vector< Well* >& wells, 
                       const std::string& name, 
                       const unsigned int& n_aquifers)
  : wells(wells),
  
    grid_create(Model_base::rect),
    down_left(Point<2>(0.0,0.0)),
    up_right(Point<2>(1.0,1.0)),
    dirichlet_function(nullptr),
    rhs_function(nullptr),
    triangulation_changed(true),
    is_adaptive(false),
    init_refinement(0),
    
    n_aquifers(n_aquifers),
    
    cycle_(-1),
    last_run_time_(0),
    solver_it(0),
    matrix_output_(false),
    sparsity_pattern_output_(false),
    output_dir("../output/model/"),
    main_output_dir("../output/"),
    name(name)
    
{
  //DBGMSG("Model_base constructor, wells_size: %d\n",this->wells.size());
  transmisivity.resize(n_aquifers, 1.0);
  //TODO: check if all wells lies in the area
  prm_declare();
}

Model_base::Model_base(const Model_base &model, std::string name)
: wells(model.wells),

  grid_create(model.grid_create),
  down_left(model.down_left),
  up_right(model.up_right),
  dirichlet_function(nullptr),
  rhs_function(nullptr),
  triangulation_changed(true),
  is_adaptive(false),
  init_refinement(model.init_refinement),
  
  n_aquifers(model.n_aquifers),
  transmisivity(model.transmisivity),
  
  cycle_(-1),
  last_run_time_(0),
  solver_it(0),
  matrix_output_(false),
  sparsity_pattern_output_(false),
  output_dir(model.main_output_dir+name+"/"),
  main_output_dir(model.main_output_dir),
  name(name)
  
{  
}

Model_base::~Model_base()
{
}


void Model_base::run(const unsigned int cycle)
{  
  cycle_++;
  if(cycle == 0)
  {
    make_grid();
    //if initial refinement is set
    /*
    if(grid_create == load || grid_create == load_circle)
    for(unsigned int r=0; r < init_refinement; r++)
      refine_grid();*/
  }
  else if (is_adaptive)
    refine_grid();

  clock_t start, stop;
  last_run_time_ = 0.0;

  /* Start timer */
  MASSERT((start = clock())!=-1, "Measure time error.");

  if (triangulation_changed == true)
    setup_system();
  assemble_system();
  solve();
 
  /* Stop timer */
  stop = clock();
  last_run_time_ = ((double) (stop-start))/CLOCKS_PER_SEC;
  std::cout << "Run time: " << last_run_time_ << " s" << std::endl;
}


void Model_base::prm_declare()
{
    parameter_handler_.enter_subsection("aquifer");
    {
        parameter_handler_.declare_entry("width","1.0",
                                        Patterns::Double(),
                                        "Width of the area.");
        parameter_handler_.declare_entry("x","0.0",
                                        Patterns::Double(),
                                        "corner_x.");
        parameter_handler_.declare_entry("y","0.0",
                                        Patterns::Double(),
                                        "corner_y.");
        parameter_handler_.declare_entry("transmisivity","1.0",
                                        Patterns::Double(),
                                        "Transmisivity of the aquifer.");
    }
    parameter_handler_.leave_subsection();
    
    parameter_handler_.declare_entry("n_wells","1",
                                     Patterns::Integer(1, 1000),
                                     "Number of wells.");
            
    parameter_handler_.enter_subsection("wells");
    {
        parameter_handler_.declare_entry("center_x","0.0",
                                        Patterns::List(Patterns::Double(),1),
                                        "X coordinates of centers of the wells.");
        parameter_handler_.declare_entry("center_y","0.0",
                                        Patterns::List(Patterns::Double(),1),
                                        "Y coordinates of centers of the wells.");
        
        parameter_handler_.declare_entry("radius","1.0",
                                        Patterns::List(Patterns::Double(),1),
                                        "Radii of wells.");
        parameter_handler_.declare_entry("perm2fer","1e5",
                                        Patterns::List(Patterns::Double(),1),
                                        "Permeability between the wells and aquifer.");
        parameter_handler_.declare_entry("perm2tard","1e5",
                                        Patterns::List(Patterns::Double(),1),
                                        "Permeability of the wells between aquitards.");
        parameter_handler_.declare_entry("pressure","1.0",
                                        Patterns::List(Patterns::Double(),1),
                                        "Pressure head in the wells.");
        
        parameter_handler_.declare_entry("n_quadrature_points","100",
                                        Patterns::Integer(10,1e5),
                                        "X coordinates of centers of the wells.");
    }
    parameter_handler_.leave_subsection();
    
    parameter_handler_.print_parameters(std::cout, ParameterHandler::Text);
    /*
    double p_a = 10.0,    //area of the model
         p_b = 10.0,
         excenter = 0,//0.61, //0.05,
         radius = p_a*std::sqrt(2),
         well_radius = 0.02,
         perm2fer = Parameters::perm2fer, 
         perm2tard = Parameters::perm2tard,
         transmisivity = Parameters::transmisivity,
         enrichment_radius = 15.0,
         well_pressure = Parameters::pressure_at_top;
         
  unsigned int n_well_q_points = 200,
               initial_refinement = 3;
         
  Point<2> well_center(0+excenter,0+excenter);
  
  std::string input_dir = "../input/square_convergence/";
  std::string coarse_file = input_dir + "coarse_grid.msh";

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
  
  //FEM model creation
  Model_simple model_simple(well);  
  model_simple.set_name(test_name + "fem_model");
  model_simple.set_output_dir(output_dir);
  model_simple.set_area(down_left,up_right);
  model_simple.set_transmisivity(transmisivity,0);
  model_simple.set_refinement(initial_refinement);  
  model_simple.set_ref_coarse_percentage(0.95,0.05);
  //model_simple.set_ref_coarse_percentage(0.3,0.05);
  //model_simple.set_grid_create_type(Model_base::rect);
  
  model_simple.set_grid_create_type(Model_base::rect);
  //model_simple.set_computational_mesh(coarse_file);
  model_simple.set_dirichlet_function(exact_solution);
  model_simple.set_adaptivity(true);
  model_simple.set_matrix_output(false);
  
  XModel_simple xmodel(well);  
//   xmodel.set_name(test_name + "sgfem_model"); 
//   xmodel.set_enrichment_method(Enrichment_method::sgfem);
  xmodel.set_name(test_name + "xfem_shift_model");
  xmodel.set_enrichment_method(Enrichment_method::xfem_shift);
  
  xmodel.set_output_dir(output_dir);
  xmodel.set_area(down_left,up_right);
  xmodel.set_transmisivity(transmisivity,0);
  xmodel.set_refinement(initial_refinement);                                     
  xmodel.set_enrichment_radius(enrichment_radius);
  xmodel.set_grid_create_type(Model_base::rect);
  xmodel.set_dirichlet_function(exact_solution);
  xmodel.set_adaptivity(true);
  //xmodel.set_well_computation_type(Well_computation::sources);
  xmodel.set_output_features(1,0,1); //decomposed, shape_func, error
  xmodel.set_matrix_output(true);
  //*/
}

void Model_base::read_input_file()
{
    MASSERT(!input_file_.empty(),"Input file has not been defined yet. Call set_input_file().");
    std::ifstream in;
    in.open(input_file_);
    
    if(in.is_open())
    {
        prm_read_inputfile(in);
    }          
    else
    {
        xprintf(Err, "Could not open input file: %s", input_file_.c_str());
    }    
    
}

void Model_base::prm_read_inputfile(ifstream& in)
{
    //reading data from filestream
    parameter_handler_.read_input(input_file_);
    
    DBGMSG("Input file read successfully.\n");
    parameter_handler_.enter_subsection("aquifer");
    {
        double w = parameter_handler_.get_double("width");
        DBGMSG("width.\n");
        down_left = Point<2>(parameter_handler_.get_double("x"),
                             parameter_handler_.get_double("y"));
        up_right = Point<2>(down_left[0] + w,
                            down_left[1] + w);
        n_aquifers = 1;
        transmisivity.resize(1);
        transmisivity[0] = parameter_handler_.get_double("transmisivity");
    }
    parameter_handler_.leave_subsection();
    
    //wells.resize(parameter_handler_.get_integer("n_wells"));
    parameter_handler_.enter_subsection("wells");
    {
        std::string temp = parameter_handler_.get("radius");
        //DBGMSG("%s\n",parameter_handler_.get("radius").c_str());
        double r = std::stod(temp);
        
        temp = parameter_handler_.get("center_x");
        double x = std::stod(temp);
        temp = parameter_handler_.get("center_x");
        double y = std::stod(temp);
        temp = parameter_handler_.get("perm2fer");
        double perm2fer = std::stod(temp);
        temp = parameter_handler_.get("perm2tard");
        double perm2tard = std::stod(temp);
        temp = parameter_handler_.get("pressure");
        double pressure = std::stod(temp);
        temp = parameter_handler_.get("n_quadrature_points");
        double n_quadrature_points = std::stod(temp);
        
        //wells[0] = new Well(r, Point<2>(x,y), perm2fer, perm2tard);
    }
    parameter_handler_.leave_subsection();
}



void Model_base::set_transmisivity(const double& trans, const unsigned int& m_aquifer)
{
  if(m_aquifer < transmisivity.size())  
    transmisivity[m_aquifer] = trans; 
  else
    xprintf(Warn,"Transmisivity not set. Size: %d, index: %d\n",transmisivity.size(), m_aquifer);
}

void Model_base::set_transmisivity(const std::vector< double >& trans)
{
  transmisivity.clear();
  transmisivity = trans;
}

void Model_base::set_area(const dealii::Point< 2 >& down_left, const dealii::Point< 2 >& up_right)
{
  MASSERT( (down_left[0] < up_right[0]) && (down_left[1] < up_right[1]), 
             "Wrong point setting - must be down left and up right vertex.");
  this->down_left = down_left; 
  this->up_right = up_right; 
}

void Model_base::set_output_dir(const std::string& path)
{
  main_output_dir = path;
  
  DIR *dir;
  /* Try to open directory */
    dir = opendir(output_dir.c_str());
    if(dir == NULL) {
        /* Directory doesn't exist. Create new one. */
        int ret = mkdir(output_dir.c_str(), 0777);

        if(ret != 0) {
            xprintf(Err, "Couldn't create directory: %s\n", output_dir.c_str());
        }
    } else {
        closedir(dir);
    }
    
    output_dir = main_output_dir + "/" + name + "/";
    
    dir = opendir(output_dir.c_str());
    if(dir == NULL) {
        /* Directory doesn't exist. Create new one. */
        int ret = mkdir(output_dir.c_str(), 0777);

        if(ret != 0) {
            xprintf(Err, "Couldn't create directory: %s\n", output_dir.c_str());
        }
    } else {
        closedir(dir);
    }
}


void Model_base::write_block_sparse_matrix(const dealii::BlockSparseMatrix< double >& matrix, const string& filename)
{
  // WHOLE SYSTEM MATRIX  ----------------------------------------------------------------
  std::string path = output_dir + filename + ".m";
  std::ofstream output (path);
  
  if(! output.is_open()) 
    xprintf(Warn, "Could not open file to write matrix: %s", path.c_str());
  
  unsigned int m = matrix.m();
  
  for(unsigned int i=0; i < m; i++)
  {
    for(unsigned int j=0; j < m; j++)
    {
      output << matrix.el(i,j) << " ";
    }
    output << "\n";
  }
  
  output.close();
  
  // A MATRIX ----------------------------------------------------------------
  path = output_dir + filename + "_a.m";
  output.clear();
  output.open(path);
  
  if(! output.is_open()) 
    xprintf(Warn, "Could not open file to write matrix: %s", path.c_str());
  
  m = matrix.block(0,0).m();
  
  for(unsigned int i=0; i < m; i++)
  {
    for(unsigned int j=0; j < m; j++)
    {
      output << matrix.block(0,0).el(i,j) << " ";
    }
    output << "\n";
  }
  
  output.close();
  
  
  // E MATRIX ----------------------------------------------------------------
  path = output_dir + filename + "_e.m";
  output.clear();
  output.open(path);
  
  if(! output.is_open()) 
    xprintf(Warn, "Could not open file to write matrix: %s", path.c_str());
  
  m = matrix.block(1,1).m();
  
  for(unsigned int i=0; i < m; i++)
  {
    for(unsigned int j=0; j < m; j++)
    {
      output << matrix.block(1,1).el(i,j) << " ";
    }
    output << "\n";
  }
  
  output.close();
}




std::pair< double, double > Model_base::integrate_difference(dealii::Vector< double >& diff_vector, const Function< 2 >& exact_solution)
{
    DBGMSG("Warning: method 'integrate_difference' needs to be implemented in descendants.\n");
}
