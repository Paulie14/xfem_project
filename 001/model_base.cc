


#include "model_base.hh"
#include "system.hh"
#include <deal.II/base/function.h>

#include <dirent.h>
#include <sys/stat.h>
#include <time.h>
#include <fstream>
#include <iostream>

Model_base::Model_base()
:   grid_create(Model_base::rect),
    down_left(Point<2>(0.0,0.0)),
    up_right(Point<2>(1.0,1.0)),
    dirichlet_function(NULL),
    triangulation_changed(true),
    is_adaptive(false),
    init_refinement(0),
    
    n_aquifers(1),
    
    last_run_time_(0),
    output_dir("../output/model/"),
    main_output_dir("../output/"),
    name("model_base")
{
}

Model_base::Model_base(const std::string& name, 
                       const unsigned int& n_aquifers)
  : grid_create(Model_base::rect),
    down_left(Point<2>(0.0,0.0)),
    up_right(Point<2>(1.0,1.0)),
    dirichlet_function(NULL),
    triangulation_changed(true),
    is_adaptive(false),
    init_refinement(0),
    
    n_aquifers(n_aquifers),
    
    last_run_time_(0),
    output_dir("../output/model/"),
    main_output_dir("../output/"),
    name(name)
    
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
    dirichlet_function(NULL),
    triangulation_changed(true),
    is_adaptive(false),
    init_refinement(0),
    
    n_aquifers(n_aquifers),
    
    last_run_time_(0),
    output_dir("../output/model/"),
    main_output_dir("../output/"),
    name(name)
    
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
  dirichlet_function(NULL),
  triangulation_changed(true),
  is_adaptive(false),
  init_refinement(model.init_refinement),
  
  n_aquifers(model.n_aquifers),
  transmisivity(model.transmisivity),
  
  output_dir(model.main_output_dir+name+"/"),
  main_output_dir(model.main_output_dir),
  name(name)
  
{  
}

Model_base::~Model_base()
{
}


Model_base::Boundary_pressure::Boundary_pressure() : Function<2>()
{
}


void Model_base::run(const unsigned int cycle)
{
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
  //std::stringstream filename1; 
  //filename1 << output_dir << "xfem_matrix";
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
}
