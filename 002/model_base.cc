


#include "model_base.hh"
#include "system.hh"
#include <deal.II/base/function.h>

#include <dirent.h>
#include <sys/stat.h>
#include <time.h>
#include <fstream>
#include <iostream>

const ModelBase::OutputOptionsType ModelBase::default_output_options_ = ModelBase::output_solution 
                                                                          | ModelBase::output_decomposed 
                                                                          | ModelBase::output_gmsh_mesh;
const unsigned int ModelBase::adaptive_integration_refinement_level_ = 12;
const unsigned int ModelBase::solver_max_iter_ = 4000;
const double ModelBase::solver_tolerance_ = 1e-12;
const double ModelBase::output_element_tolerance_ = 1e-3;


ModelBase::ModelBase()
:   grid_create(ModelBase::rect),
    down_left(Point<2>(0.0,0.0)),
    up_right(Point<2>(1.0,1.0)),
    dirichlet_function(nullptr),
    rhs_function(nullptr),
    triangulation_changed(true),
    is_adaptive(false),
    initial_refinement_(0),
    
    n_aquifers_(1),
    
    cycle_(-1),
    last_run_time_(0),
    solver_iterations_(0),
    output_dir_("../output/model/"),
    main_output_dir_("../output/"),
    name_("model_base"),
    output_options_(default_output_options_)
{
}

ModelBase::ModelBase(const std::string& name, 
                       const unsigned int& n_aquifers)
  : grid_create(ModelBase::rect),
    down_left(Point<2>(0.0,0.0)),
    up_right(Point<2>(1.0,1.0)),
    dirichlet_function(nullptr),
    rhs_function(nullptr),
    triangulation_changed(true),
    is_adaptive(false),
    initial_refinement_(0),
    
    n_aquifers_(n_aquifers),
    
    cycle_(-1),
    last_run_time_(0),
    solver_iterations_(0),
    output_dir_("../output/model/"),
    main_output_dir_("../output/"),
    name_(name),
    output_options_(default_output_options_)
    
{
  transmisivity.resize(n_aquifers_, 1.0);
}

ModelBase::ModelBase(const std::vector< Well* >& wells, 
                       const std::string& name, 
                       const unsigned int& n_aquifers)
  : wells(wells),
  
    grid_create(ModelBase::rect),
    down_left(Point<2>(0.0,0.0)),
    up_right(Point<2>(1.0,1.0)),
    dirichlet_function(nullptr),
    rhs_function(nullptr),
    triangulation_changed(true),
    is_adaptive(false),
    initial_refinement_(0),
    
    n_aquifers_(n_aquifers),
    cycle_(-1),
    last_run_time_(0),
    solver_iterations_(0),

    output_dir_("../output/model/"),
    main_output_dir_("../output/"),
    name_(name),
    output_options_(default_output_options_)
    
{
  //DBGMSG("ModelBase constructor, wells_size: %d\n",this->wells.size());
  transmisivity.resize(n_aquifers, 1.0);
  //TODO: check if all wells lies in the area
}

ModelBase::ModelBase(const ModelBase &model, std::string name)
: wells(model.wells),

  grid_create(model.grid_create),
  down_left(model.down_left),
  up_right(model.up_right),
  dirichlet_function(nullptr),
  rhs_function(nullptr),
  triangulation_changed(true),
  is_adaptive(false),
  initial_refinement_(model.initial_refinement()),
  
  n_aquifers_(model.n_aquifers()),
  transmisivity(model.transmisivity),
  
  cycle_(-1),
  last_run_time_(0),
  solver_iterations_(0),
  output_dir_(model.main_output_dir_+name+"/"),
  main_output_dir_(model.main_output_dir_),
  name_(name),
  output_options_(default_output_options_)
  
{  
}

ModelBase::~ModelBase()
{
}


void ModelBase::run(const unsigned int cycle)
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



void ModelBase::set_transmisivity(const double& trans, const unsigned int& m_aquifer)
{
  if(m_aquifer < transmisivity.size())  
    transmisivity[m_aquifer] = trans; 
  else
    xprintf(Warn,"Transmisivity not set. Size: %d, index: %d\n",transmisivity.size(), m_aquifer);
}

void ModelBase::set_transmisivity(const std::vector< double >& trans)
{
  transmisivity.clear();
  transmisivity = trans;
}

void ModelBase::set_area(const dealii::Point< 2 >& down_left, const dealii::Point< 2 >& up_right)
{
  MASSERT( (down_left[0] < up_right[0]) && (down_left[1] < up_right[1]), 
             "Wrong point setting - must be down left and up right vertex.");
  this->down_left = down_left; 
  this->up_right = up_right; 
}

void ModelBase::set_output_dir(const std::string& path)
{
  main_output_dir_ = path;
  
  DIR *dir;
  /* Try to open directory */
    dir = opendir(output_dir_.c_str());
    if(dir == NULL) {
        /* Directory doesn't exist. Create new one. */
        int ret = mkdir(output_dir_.c_str(), 0777);

        if(ret != 0) {
            xprintf(Err, "Couldn't create directory: %s\n", output_dir_.c_str());
        }
    } else {
        closedir(dir);
    }
    
    output_dir_ = main_output_dir_ + "/" + name_ + "/";
    
    dir = opendir(output_dir_.c_str());
    if(dir == NULL) {
        /* Directory doesn't exist. Create new one. */
        int ret = mkdir(output_dir_.c_str(), 0777);

        if(ret != 0) {
            xprintf(Err, "Couldn't create directory: %s\n", output_dir_.c_str());
        }
    } else {
        closedir(dir);
    }
}


void ModelBase::write_block_sparse_matrix(const dealii::BlockSparseMatrix< double >& matrix, const string& filename)
{
  // WHOLE SYSTEM MATRIX  ----------------------------------------------------------------
  std::string path = output_dir_ + filename + ".m";
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
  path = output_dir_ + filename + "_a.m";
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
  path = output_dir_ + filename + "_e.m";
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




std::pair< double, double > ModelBase::integrate_difference(dealii::Vector< double >& diff_vector, const Function< 2 >& exact_solution)
{
    MASSERT(0,"Warning: method 'integrate_difference' needs to be implemented in descendants.\n");
    return std::make_pair<double, double>(0,0);
}
