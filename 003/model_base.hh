#ifndef ModelBase_h
#define ModelBase_h

#include <deal.II/grid/tria.h>
#include <deal.II/base/function.h>

//Forward declaration
class Well;
namespace compare {
    class ExactBase;
}
namespace dealii {
    template<int> class Function;
    template<typename> class BlockSparseMatrix;
}
class GlobalSettingWriter;

/** @brief Base class for the model of multi-aquifer system with wells.
 * 
 * This is the base class for multi-aquifer model with wells. 
 * Here the simple rutines of setting and getting parameters are implemented.
 */
class ModelBase
{
public:
  ///type of grid creation
  typedef enum { 
    rect,   //for rectangular grid
    circle, //for circular grid
    load,    //for loaded grid from file   
    load_circle //for loading flags and recreating circle mesh
  } grid_create_type;
  
  typedef unsigned int OutputOptionsType;
  ///type of grid creation
  enum OutputOptions {
    output_solution = 0x001,
    output_decomposed = 0x002,
    output_error = 0x004,
    output_matrix = 0x008,
    output_adaptive_plot = 0x010,
    output_gmsh_mesh = 0x020,
    output_vtk_mesh = 0x040,
    output_sparsity_pattern = 0x080,
    output_shape_functions = 0x100,
    //other free ...
    output_all = 0x800
  }; 
  
  ///Default constructor.
  ModelBase();
  
  ///Constructor.
  /**
   * @param name is the name of model (creates output directory with this name)
   * @param n_aquifers is the number of aquifers (not used sofar)
   */
  ModelBase(const std::string &name, 
            const unsigned int &n_aquifers=1);
  
  ///Constructor.
  /**
   * @param wells is the vector of pointers to wells.
   * @param name is the name of model (creates output directory with this name)
   * @param n_aquifers is the number of aquifers (not used sofar)
   */
  ModelBase(const std::vector<Well*> &wells,
            const std::string &name, 
            const unsigned int &n_aquifers=1);
  
  ///Kind of copy constructor.
  /**
   * @param model is another model to be copied
   * @param name is a new name for the model   
   */
  ModelBase(const ModelBase &model, std::string name);
  
  /// Destructor
  virtual ~ModelBase ();
    
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
                                             
  virtual std::pair<double,double> integrate_difference(dealii::Vector<double>& diff_vector, 
                                                        compare::ExactBase *exact_solution, bool h1=false);
  
                                             
  /** @name Getters
   */
  //@{
                                             
  /** Returns constant reference to the computed solution.
     */
  virtual const dealii::Vector< double > &get_solution() = 0;
  
  /** Returns constant reference to computed distributed solution (on different grid).
     */
  virtual const dealii::Vector< double > &get_distributed_solution() = 0;
    
  ///Gets support points of the model.
  ///So far the support points corresponds with nodes
 // virtual void get_support_points (std::vector<Point<2> > &support_points) = 0;
  
  ///Getter of initial refinement.
  unsigned int initial_refinement() const;
  
  ///Getter of transmisivity
  ///@param m_aquifer is the number aquifer.
  double transmisivity(const unsigned int &m_aquifer) const;
  
  /// Returns model name.
  std::string name() const;
  
  /// Returns cycle number.
  unsigned int cycle() const;
  
  /// Returns number of iterations
  unsigned int solver_iterations() const;
  
  /// Returns time of the last run - includes only methods @p setup, @p assemble and @p solve.
  double last_run_time() const;
  
  /// Returns number of aquifers.
  unsigned int n_aquifers() const;
  
  /// Returns path to the output directory.
  std::string output_dir() const;
  //@}
  
  /** @name Setters
   */
  //@{
  ///Setter of area dimensions
  ///@param down_left is down left vertex of rectangle area
  ///@param up_right is up right vertex of rectangle area
  void set_area(const dealii::Point<2> &down_left, const dealii::Point<2> &up_right);
  
  ///Setter of transmisivity
  ///@param trans is transmisivity to be set for \f$ m \f$-th aquifer
  ///@param m_aquifer is the number aquifer.
  void set_transmisivity(const double &trans, const unsigned int &m_aquifer);
    
  ///Setter of transmisivity
  ///@param trans is a vector of transmisivities of all aquifers
  void set_transmisivity(const std::vector<double> &trans);
  
  ///Setter of initial refinement level. Default is 0.
  ///@param ref initial refinement level to be set
  void set_initial_refinement(unsigned int ref);
  
  ///Setter of the main output directory.
  ///@param path file path a directory
  void set_output_dir(const std::string &path);
  
  ///Sets the name of the model and creates individual output directory. Default is 'model_base'.
  ///@param name is the name to be set
  void set_name(const std::string &name);
  
  ///Sets how the grid should be created. Default is 'rect'.
  void set_grid_create_type(grid_create_type type);
  
  ///Sets adaptivity on and off. Default is false.
  void set_adaptivity (bool is_on);

  ///Sets the vector of wells.
  void set_wells(const std::vector<Well*> &wells);
  
  ///Sets the function defining the Dirichlet function.
  void set_dirichlet_function(dealii::Function<2> *func);
  
  ///Sets the function defining the Dirichlet function.
  void set_rhs_function(dealii::Function<2> *func);
  
  void set_output_options(OutputOptionsType output_options);
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
  virtual void assemble_dirichlet(unsigned int m) 
  {std::cout << "No Dirichlet BC is applied." << std::endl; };

  //@}
  
  //Writes matrix in matlab .m file.
  void write_block_sparse_matrix(const dealii::BlockSparseMatrix<double> &matrix, 
                                 const std::string &filename);
  
  ///Vector of wells.
  std::vector<Well*> wells;
    
  ///type of grid creation
  grid_create_type grid_create;
  
  
  dealii::Point<2> down_left, ///< down left corner point
                   up_right;  ///< up right corner point
  
  ///Pointer to function describing Dirichlet boundary condition.
  dealii::Function<2> *dirichlet_function;         
  ///Pointer to function describing RHS - sources.
  dealii::Function<2> *rhs_function;
  
  ///flag is true if the triangulation has been changed in the current cycle
  bool triangulation_changed;
  
  ///do we want adaptive refinement in each cycle
  bool is_adaptive;
  
  ///initial refinement of the grid
  unsigned int initial_refinement_;
 
  ///number of aquifers
  unsigned int n_aquifers_;
  
  ///transmisivity of the aquifer
  std::vector<double> transmisivity_;

  ///@name System
  //@{
  unsigned int cycle_;
  ///last run time (setup, assemble, solve)
  double last_run_time_;
  
  unsigned int solver_iterations_;
  
  /** @brief Path to output directory.
   * Name should be defined before, else name="Model".
   * The path is set to "main_output_dir/output_dir"
   */
  std::string output_dir_;
  ///path to the main output directory
  std::string main_output_dir_;
  
  ///name of model (name of created output directory)
  std::string name_;
  
  //@}
  
  ///@name Output Options
  //@{
    OutputOptionsType output_options_;
  
    static const OutputOptionsType default_output_options_;
    static const unsigned int adaptive_integration_refinement_level_;
    static const unsigned int solver_max_iter_;
    static const double solver_tolerance_;
    static const double output_element_tolerance_;
  //@}
    
  friend class GlobalSettingWriter;
};





/*******************************************     IMPLEMENTATION                   ***************************/


  inline unsigned int ModelBase::initial_refinement() const
  {return initial_refinement_;}

  inline double ModelBase::transmisivity(const unsigned int &m_aquifer) const
  {return transmisivity_[m_aquifer];}
  
  inline std::string ModelBase::name() const
  { return name_; }
  
  inline unsigned int ModelBase::cycle() const
  { return cycle_; }
  
  inline unsigned int ModelBase::solver_iterations() const
  {return solver_iterations_; }

  inline double ModelBase::last_run_time() const
  { return last_run_time_;}
  
  inline unsigned int ModelBase::n_aquifers() const
  { return n_aquifers_; }
  
  inline std::string ModelBase::output_dir() const
  { return output_dir_; }
  
  
  inline void ModelBase::set_initial_refinement(unsigned int ref)
  {initial_refinement_ = ref;}
  
  inline void ModelBase::set_name(const std::string &name)
  { this->name_ = name;
    set_output_dir(main_output_dir_); //reseting output directory
  }
  
  inline void ModelBase::set_grid_create_type(grid_create_type type)
  { grid_create = type; }
  
  inline void ModelBase::set_adaptivity (bool is_on)
  { this->is_adaptive = is_on;}

  inline void ModelBase::set_wells(const std::vector<Well*> &wells)
  { this->wells.clear();
    this->wells = wells;
  }
  
  inline void ModelBase::set_dirichlet_function(dealii::Function<2> *func)
  { dirichlet_function = func; }
  
  inline void ModelBase::set_rhs_function(dealii::Function<2> *func)
  { rhs_function = func; }
  
  inline void ModelBase::set_output_options(OutputOptionsType output_options)
  { output_options_ = output_options;}

#endif  //ModelBase_h
