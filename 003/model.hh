#ifndef Model_h
#define Model_h

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_accessor.h>
 
#include <deal.II/base/quadrature_lib.h>

//block things
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include "model_base.hh"

using namespace dealii;

//forward declarations
namespace dealii{
    template<int, int> class PersistentTriangulation;
    template<int,int> class DoFHandler;
}

namespace Solution{ class ExactBase; }
class Well;
class DataCell;

/// class Model
/**
 * @brief Model of an aquifer with wells using classical FEM on an adaptively refined grid.
 * Homogenous Neumann boundary condition is considered on the areas boundary.
 * The model may be run in cycles to compute adaptively refined grid.
 */
class Model : public ModelBase
{
  public:
    ///@brief Default constructor.
    /// The Model cannot be run from the state after creation.
    Model ();
    
    ///@brief Constructor.
    Model (const std::string &name,
           const unsigned int &n_aquifers);
    
    ///@brief Constructor.
    Model (const std::vector<Well*> &wells, 
           const std::string &name="Adaptive_FEM",
           const unsigned int &n_aquifers=1);
    
    virtual ~Model();
    
    ///@name Output
    //@{
    //output results from another model but computed on the model mesh
    void output_foreign_results(const unsigned int cycle,const Vector<double> &foreign_solution);
    
    
    void output_results (const unsigned int cycle=0) override;
    
    void output_distributed_solution ( const std::string &mesh_file,
                                       const std::string &flag_file,
                                       bool is_circle=false,
                                       const unsigned int &cycle=0, 
                                       const unsigned int &m_aquifer=0);
    
    void output_distributed_solution(const dealii::Triangulation< 2 >& dist_tria, 
                                     const unsigned int& cycle);
    
    std::pair<double, double> integrate_difference(Vector<double>& diff_vector, 
                                                   Solution::ExactBase * exact_solution, 
                                                   bool h1=false) override;
    //@}
                                                   
    ///@name Getters
    //@{
    /** Returns constant reference to distributed solution.
     */
    const Vector< double > &get_solution() override;
    
    /** Returns constant reference to distributed solution.
     */
    const Vector< double > &get_distributed_solution() override;
    
    ///Returns pointer to the computational triangulation (PersistentTriangulation).
    const Triangulation<2> &get_triangulation();
    
    ///Returns number of degrees of freedom
    unsigned int get_number_of_dofs();
    //@}
    
    ///@name Setters
    //@{
    /** @brief Sets the percentage of refinement and coarsening for Kelly Estimator.
      * @param ref is percentage of refinement
      * @param coarse is percentage of coarsening
      */
    void set_ref_coarse_percentage(float ref, float coarse);

    /**Sets file path to a mesh file. 
     * Grid creation type @p grid_create is set to @p load.
     * @param coarse_mesh is the path to the GMSH file to be loaded
     * @param ref_flags is the path to the file with refinement flags
     */
    void set_computational_mesh (std::string coarse_mesh, std::string ref_flags = "");
    
    /**Sets file path to a mesh file. 
     * Grid creation type @p grid_create is set to @p load_circle.
     * @param ref_flags is the path to the file with refinement flags
     * @param center is center of the circle
     * @param radius is the radius of the circle
     */
    void set_computational_mesh_circle(std::string ref_flags, 
                                       Point<2> center, 
                                       double radius);
    //@}

  protected:
    void make_grid () override;
    virtual void refine_grid() override;
    void setup_system () override;
    void assemble_system () override;
    void assemble_dirichlet (unsigned int m) override;
    void solve () override;

    ///Function finding the cells through which the well boundary goes.
    ///Uses recusively add_points_to_cell() inside.
    void find_well_cells();
    
    ///Recursively used function for adding quadrature points of the well to their cells.
    /** 
     * Iterates over all quadrature points of the well and checks if they lie in the cell.
     * If no point lies in the cell then returns.
     * Otherwise goes over all neighbors and calls self recusively.
     * @param cell is the cell which we are testing.
     * @param well is the well which boundary we are testing. We test only one well at time. 
     */
    void add_data_to_cell(const DoFHandler<2>::active_cell_iterator cell, 
                          Well *well, 
                          unsigned int well_index);

    
    ///path to computational mesh
    std::string coarse_grid_file;
    ///path to refinement flag file
    std::string ref_flags_file;
    
    ///center of the circle area of computation
    Point<2> center;
    ///radius of the circle area of computation
    double radius;
    
    ///2d triangulation of an aquifer
    PersistentTriangulation<2> *triangulation;
    ///coarse 2d triangulation, used by triangulation
    Triangulation<2>            coarse_tria;
    
    ///percentage of elements that should be refined
    float refinement_percentage;
    ///percentage of elements that should be coarsen
    float coarsing_percentage;
    
    ///Cell data carrying information about wells and qudrature points on their edges
    ///relating to a specific cell of triangulation
    std::vector<DataCell*> data_cell;
    
    /** vector holding number of quadrature points on hte edge pf each well
     * is used for correting integral weights 
     * (if the point lies exactly on the edge of cell, then it is computed twice)
     */
    std::vector<unsigned int> n_wells_q_points;
    
    //must be in this order, else a desctructor is needed to call dof_handler.clear();
    ///2d polynomial finite element
    FE_Q<2>              fe;                    
    ///degrees of freedom handler
    DoFHandler<2>*       dof_handler;
    
    ///Quadrature for integrating on elements
    QGauss<2>  quadrature_formula;
    
    
    ///matrix containing constraints coming from hanging nodes
    ConstraintMatrix hanging_node_constraints;

    ///sparsity pattern for system matrix
    BlockSparsityPattern        block_sp_pattern;
    ///system matrix
    BlockSparseMatrix<double>   block_matrix;   
    
    ///vector of solution
    BlockVector<double> block_solution;
    ///vector of right hand side
    BlockVector<double> block_system_rhs;
    
    Vector<double>      dist_solution;
};


/****************************************            Implementation          ********************************/

inline const Triangulation<2> & Model::get_triangulation()
{ return *triangulation; }

inline unsigned int Model::get_number_of_dofs()
{ return dof_handler->n_dofs(); }
    
inline void Model::set_ref_coarse_percentage(float ref, float coarse)
{ 
  refinement_percentage = ref;
  coarsing_percentage = coarse;
}

inline void Model::set_computational_mesh (std::string coarse_mesh, std::string ref_flags)
{ this->coarse_grid_file = coarse_mesh;
  this->ref_flags_file = ref_flags;
  grid_create = load;
}

inline void Model::set_computational_mesh_circle(std::string ref_flags, 
                                          Point<2> center, 
                                          double radius)
{
  this->center = center;
  this->radius = radius;
  grid_create = load_circle;
  this->ref_flags_file = ref_flags;
}
#endif  //Model_h