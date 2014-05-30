#ifndef Model_base_h
#define Model_base_h

#include <deal.II/grid/tria.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/block_sparse_matrix.h>


using namespace dealii;

//Forward declaration
class Well;

/** @brief Base class for the model of multi-aquifer system with wells.
 * 
 * This is the base class for multi-aquifer model with wells. 
 * Here the simple rutines of setting and getting parameters are implemented.
 */
class Model_base
{
public:
  ///type of grid creation
  typedef enum { 
    rect,   //for rectangular grid
    circle, //for circular grid
    load,    //for loaded grid from file   
    load_circle //for loading flags and recreating circle mesh
  } grid_create_type;
  
  ///Default constructor.
  Model_base();
  
  ///Constructor.
  /**
   * @param name is the name of model (creates output directory with this name)
   * @param n_aquifers is the number of aquifers (not used sofar)
   */
  Model_base(const std::string &name, 
             const unsigned int &n_aquifers=1);
  
  ///Constructor.
  /**
   * @param wells is the vector of pointers to wells.
   * @param name is the name of model (creates output directory with this name)
   * @param n_aquifers is the number of aquifers (not used sofar)
   */
  Model_base(const std::vector<Well*> &wells,
             const std::string &name, 
             const unsigned int &n_aquifers=1);
  
  ///Kind of copy constructor.
  /**
   * @param model is another model to be copied
   * @param name is a new name for the model   
   */
  Model_base(const Model_base &model, std::string name);
  
  /// Destructor
  virtual ~Model_base ();
    
  /** @brief Runs all the computations procedures.
   * 
   * This method is not pure virtual and is implemented,
   * cause all the models have to:
   * - make a grid
   * - can possibly refine the grid
   * - setup the system (prepare matrices, vectors etc.)
   * - assemble the system (computes the integrals etc.)
   * - solve the linear system
   * 
   * Outputing results is optional in every model, 
   * cause sometimes we want to output the results on different grids,
   * or we do not even want to output the results in some file format.
   */
  virtual void run (const unsigned int cycle=0);
    
  ///Pure virtual. Outputs the results on the computational grid.
  ///@param cycle number of current cycle
  virtual void output_results (const unsigned int cycle=0) = 0;
  
  /** @brief Output of a single aquifer.
   * Computes solution on the mesh loaded from mesh file of the same dimensions as the 
   * model was computed.
   * @param mesh_file is the path to the file with mesh [GMSH format].
   * @param flag_file is the path to the file with refinement flags (is treated as optional in models).
   * @param is_circle is true if the grid to be loaded is circular one
   * @param cycle is a number of that can be used for multiple output.
   * @param m_aquifer is the number aquifer which is to be written.
   */
  virtual void output_distributed_solution ( const std::string &mesh_file, 
                                             const std::string &flag_file,
                                             bool is_circle=false,
                                             const unsigned int &cycle=0, 
                                             const unsigned int &m_aquifer=0) = 0;
                                             
  virtual std::pair<double,double> integrate_difference(Vector<double>& diff_vector, const Function<2> &exact_solution);
  
                                             
  /** @name Getters
   */
  //@{
  ///Returns time of the last run - includes only methods @p setup, @p assemble and @p solve.
  inline double get_last_run_time()
  { return last_run_time_;}
                                             
  /** Returns constant reference to the computed solution.
     */
  virtual const Vector< double > &get_solution() = 0;
  
  /** Returns constant reference to computed distributed solution (on different grid).
     */
  virtual const Vector< double > &get_distributed_solution() = 0;
    
  ///Gets support points of the model.
  ///So far the support points corresponds with nodes
 // virtual void get_support_points (std::vector<Point<2> > &support_points) = 0;
  
  ///Getter of initial refinement.
  inline unsigned int get_refinement() const
  {return init_refinement;}
  
  ///Getter of transmisivity
  ///@param m_aquifer is the number aquifer.
  inline double get_transmisivity(const unsigned int &m_aquifer) const
  {return transmisivity[m_aquifer];}
  
  
  ///Returns number of iterations
  inline unsigned int solver_iterations() const
  {return solver_it; }
  
  //@}
  
  /** @name Setters
   */
  //@{
  ///Setter of area dimensions
  ///@param down_left is down left vertex of rectangle area
  ///@param up_right is up right vertex of rectangle area
  void set_area(const Point<2> &down_left, const Point<2> &up_right);
  
  ///Setter of transmisivity
  ///@param trans is transmisivity to be set for \f$ m \f$-th aquifer
  ///@param m_aquifer is the number aquifer.
  void set_transmisivity(const double &trans, const unsigned int &m_aquifer);
    
  ///Setter of transmisivity
  ///@param trans is a vector of transmisivities of all aquifers
  void set_transmisivity(const std::vector<double> &trans);
  
  ///Setter of initial refinement level. Default is 0.
  ///@param ref initial refinement level to be set
  inline void set_refinement(const double &ref)
  {init_refinement = ref;}
  
  ///Setter of the main output directory.
  ///@param path file path a directory
  void set_output_dir(const std::string &path);
  
  ///Sets the name of the model and creates individual output directory. Default is 'model_base'.
  ///@param name is the name to be set
  inline void set_name(const std::string &name)
  { this->name = name;
    set_output_dir(main_output_dir); //reseting output directory
  }
  
  ///Sets how the grid should be created. Default is 'rect'.
  void set_grid_create_type(grid_create_type type)
  { grid_create = type; }
  
  ///Sets adaptivity on and off. Default is false.
  inline void set_adaptivity (bool is_on)
  { this->is_adaptive = is_on;}

  ///Sets the vector of wells.
  inline void set_wells(const std::vector<Well*> &wells)
  { 
    this->wells.clear();
    this->wells = wells;
  }
  
  ///Sets the function defining the Dirichlet function.
  inline void set_dirichlet_function(Function<2> *func)
  {
    dirichlet_function = func;
  }
  
  ///Sets the function defining the Dirichlet function.
  inline void set_rhs_function(Function<2> *func)
  {
    rhs_function = func;
  }
  
  ///Sets output of the system matrix on/off. Default is false.
  inline void set_matrix_output(bool matrix_output)
  { matrix_output_ = matrix_output;}
  
  ///Sets output of the sparsity pattern on/off. Default is false.
  inline void set_sparsity_pattern_output(bool sparsity_pattern_output)
  { sparsity_pattern_output_ = sparsity_pattern_output;}
  //@}
    
    
protected:
    
  /** @name Run methods 
   * Methods called in @p run.
   */
  //@{ 
  ///Pure virtual. Is used to create/load grid.
  virtual void make_grid () = 0;
  ///Pure virtual. Is used to refine grid.
  virtual void refine_grid() = 0;
  ///Pure virtual. Is usde to setup the system of linear equations.
  virtual void setup_system () = 0;
  ///Pure virtual. Is used to assemble the elements of linear system.
  virtual void assemble_system () = 0;
  ///Pure virtual. Is used to solve the linear system.
  virtual void solve () = 0;
  
  /**This method does nothing here. 
   * It can be overriden to assemble Dirichlet boundary condition.
   * This cannot be done the same for all models, if we will consider enrichment at the boundary in the future.
   */
  virtual void assemble_dirichlet() 
  {std::cout << "No Dirichlet BC is applied." << std::endl; };

  //@}
  
  //Writes matrix in matlab .m file.
  void write_block_sparse_matrix(const BlockSparseMatrix<double> &matrix, const std::string &filename);
  
  ///Vector of wells.
  std::vector<Well*> wells;
    
  ///type of grid creation
  grid_create_type grid_create;
  
  
  Point<2> down_left, ///< down left corner point
           up_right;  ///< up right corner point
  
  ///Pointer to function describing Dirichlet boundary condition.
  Function<2> *dirichlet_function;         
  ///Pointer to function describing RHS - sources.
  Function<2> *rhs_function;
  
  ///flag is true if the triangulation has been changed in the current cycle
  bool triangulation_changed;
  
  ///do we want adaptive refinement in each cycle
  bool is_adaptive;
  
  ///initial refinement of the grid
  unsigned int init_refinement;
 
  ///number of aquifers
  unsigned int n_aquifers;
  
  ///transmisivity of the aquifer
  std::vector<double> transmisivity;

  ///@name System
  //@{
  unsigned int cycle_;
  ///last run time (setup, assemble, solve)
  double last_run_time_;
  
  unsigned int solver_it;
  bool matrix_output_,
       sparsity_pattern_output_;
  
  /** @brief Path to output directory.
   * Name should be defined before, else name="Model".
   * The path is set to "main_output_dir/output_dir"
   */
  std::string output_dir;
  ///path to the main output directory
  std::string main_output_dir;
  
  ///name of model (name of created output directory)
  std::string name;
  //@}
           
};

#endif  //Model_base_h
