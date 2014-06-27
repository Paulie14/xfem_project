


#include "model_base.hh"
#include "system.hh"
#include <deal.II/base/function.h>

#include <dirent.h>
#include <sys/stat.h>
#include <time.h>
#include <fstream>
#include <iostream>

const Model_base::OutputOptionsType Model_base::default_output_options_ = Model_base::output_solution 
                                                                          | Model_base::output_decomposed 
                                                                          | Model_base::output_gmsh_mesh;
const unsigned int Model_base::adaptive_integration_refinement_level_ = 12;
const unsigned int Model_base::solver_max_iter_ = 4000;
const double Model_base::solver_tolerance_ = 1e-12;
const double Model_base::output_element_tolerance_ = 1e-3;


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
    output_dir("../output/model/"),
    main_output_dir("../output/"),
    name("model_base"),
    output_options_(default_output_options_)
{
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
    output_dir("../output/model/"),
    main_output_dir("../output/"),
    name(name),
    output_options_(default_output_options_)
    
{
  transmisivity.resize(n_aquifers, 1.0);
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

    output_dir("../output/model/"),
    main_output_dir("../output/"),
    name(name),
    output_options_(default_output_options_)
    
{
  //DBGMSG("Model_base constructor, wells_size: %d\n",this->wells.size());
  transmisivity.resize(n_aquifers, 1.0);
  //TODO: check if all wells lies in the area
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
  output_dir(model.main_output_dir+name+"/"),
  main_output_dir(model.main_output_dir),
  name(name),
  output_options_(default_output_options_)
  
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
    MASSERT(0,"Warning: method 'integrate_difference' needs to be implemented in descendants.\n");
    return std::make_pair<double, double>(0,0);
}
