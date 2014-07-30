#ifndef BeModel_h
#define BeModel_h

#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>

#include "model_base.hh"


using namespace dealii;

class Well;

/** @brief Class representing the model computed by BEM (Boundary Element Method)
 * 
 * Implements the computation of a simple model -- Laplace equation \f$ \Delta h = 0 \f$.
 * We are solving equation
 * \f{equation}
 *     c(P)u(P) + \int\limits_\Gamma u\frac{\partial\omega}{\partial n}\, \mathrm{d}\Gamma = \int\limits_\Gamma \frac{\partial u}{\partial n}\, \omega \, \mathrm{d}\Gamma,
 * \f}
 * which we discretize into
 * \f{equation}
 *   c_iu_i + \sum\limits_{j=1}^N \sum\limits_{k=1}^{N_j} \alpha_{j,k} \int\limits_{\Gamma_j} \varphi_k \frac{\partial\omega_i}{\partial n}\, \mathrm{d}\Gamma = \sum\limits_{j=1}^N \sum\limits_{k=1}^{N_j} \beta_{j,k} \int\limits_{\Gamma_j} \varphi_k \omega_i \, \mathrm{d}\Gamma.
 * \f}
 * After we compute solution in 1D on the boundary, we can compute solution in any point \f$ P \f$ in the area by
 * \f{equation}
 *    u(P) = \sum\limits_{j=1}^N \sum\limits_{k=1}^{N_j} \beta_{j,k} \int\limits_{\Gamma_j} \varphi_k \omega(P) \,\mathrm{d}\Gamma - \sum\limits_{j=1}^N \sum\limits_{k=1}^{N_j} \alpha_{j,k} \int\limits_\Gamma \varphi_k \frac{\partial\omega(P)}{\partial n} \, \mathrm{d}\Gamma.
 * \f}
 * 
 * TODO: Find the error that causes wrong solution.
 */
class BemModel : public ModelBase
{
  public:
    ///@brief Default constructor.
    /// The Model cannot be run from the state after creation.
    BemModel ();
    
    ///@brief Constructor.
    /**Takes only a name as argument.
     * @param name is the name of the model (used during output)
     * @param n_aquifers number of aquifers in the model (not used sofar)
     */
    BemModel (const std::string &name="BEM_model",
              const unsigned int &n_aquifers=1);
    
    ///@brief Constructor.
    /** Takes name and Well object as arguments.
     * @param wells is vector of of Well objects.
     * @param name is the name of the model (used during output)
     * @param n_aquifers number of aquifers in the model (not used sofar)
     */
    BemModel (const std::vector<Well*> &wells, 
              const std::string &name="BEM_model",
              const unsigned int &n_aquifers=1);
    
    /// @brief Virtual destructor.
    ///Does nothing but the destructor of the ancestor is called after.
    virtual ~BemModel()
    {}
    
    /** @brief Run the computation of the model.
     * Prepare, assemble and solve the system.
     * Computes bem solution for specified bem_mesh.
     * @param cycle number used when calling in cycles
     */
    virtual void run (const unsigned int cycle=0);
    
    /** @brief Distributes 1D computed solution to 2D grid and outputs.
     * Output on 2d domain.
     * Computes solution on the mesh loaded from mesh file.
     * @param mesh_file mesh on which we want to evaluate the result
     * @param flag_file is file with refinement flags
     * @param cycle number used when calling in cycles
     * @param m_aquifer number of aquifer
     */
    virtual void output_distributed_solution (const std::string &mesh_file, 
                                              const std::string &flag_file="",
                                              const unsigned int &cycle=0, 
                                              const unsigned int &m_aquifer=0);
    
    /** Returns constant reference to distributed solution.
     */
    virtual const Vector< double > &get_solution();
    
    /** Returns constant reference to distributed solution.
     */
    virtual const Vector< double > &get_distributed_solution();
    
    /** @brief Computes solution in given point in the area of the model. 
     * Computes solution at specified points.
     * (e.g. support points of the model in model.h 
     * so that the two solutions can be compared)
     * @param points are the points in which the solution should be computed
     * @param solution
     */
    virtual void get_solution_at_points(const std::vector<Point<2> > &points, Vector<double> &solution);
    
    /** @brief Gets the 1D solution on the boundary.
     * @param points are the points on the boundary in which the solution should be computed
     */
    Vector<double> get_boundary_solution(const std::vector<Point<2> > &points);
    
    /** @brief Debug function.
     * Writes the BEM solution to standard output.
     * */
    void write_bem_solution();
    
    ///Sets the boundary mesh.
    ///@param mesh_file_name is the name of the file with mesh
    inline void set_bem_mesh(const std::string &mesh_file_name)
    { bem_mesh_file = mesh_file_name; }
    
    ///Overrides setting of adaptivity. Is not used in this model.
    inline void set_adaptivity (bool is_on)
    { std::cout << "Warn.: Not used in XFEM." << std::endl;}

  private:
    
    ///Returns indexes of elements on the boundary, on which the points lie.
    ///@param points are points which we want to know on which element they lie
    ///@param indexes output vector of indexes
    void get_boundary_elm_index(const std::vector<dealii::Point<2 > > & points, 
                               std::vector<unsigned int> & indexes);
    
    ///Returns value of omega - log weight function.
    ///@param R is the vector pointing from singularity to the current point
    double omega_function(const Point<2> &R);
    ///Returns value of normal derivate of omega (d(omega)/dn = nabla(omega).n)
    ///@param R is the vector pointing from singularity to the current point
    Point<2> omega_normal(const Point<2> &R);
    
    ///Quadrature factory - returns created quadrture (used to generate singular quadrature).
    const Quadrature<1> & get_singular_quadrature(
         const typename DoFHandler<1, 2>::active_cell_iterator &cell,
         const unsigned int index) const;
         
    ///Returns true if the point is in the corner of the square area.
    bool is_corner_point(Point<2> point);
    
    
    virtual void make_grid ();
    virtual void setup_system ();
    virtual void assemble_system ();
    virtual void solve ();
    virtual void output_results (const unsigned int cycle=0) ;

                
    ///Vector of wells.
    std::vector<Well*> wells;
    unsigned int number_of_wells;
    unsigned int number_of_well_elements;
    
    ///Name of file with the boundary mesh.
    std::string bem_mesh_file;
  
    ///Triangulation object - dimension=1 (boundary), around the area - spacedimension=2
    Triangulation<1,2> triangulation;
    
    //must be in this order, else a desctructor is needed to call dof_handler.clear();
    ///Finite element (dim, spacedim corresponds with triangulation)
    FE_Q<1,2>           fe;                    
    ///Dof handler (dim, spacedim corresponds with triangulation)
    DoFHandler<1,2>     dof_handler;
    ///Element mapping (dim, spacedim corresponds with triangulation)
    MappingQ<1,2>       mapping;
    
    ///Quadrature for regular integral on elements.
    QGauss<1>     quadrature_formula;
    
    ///Order of singular quadrature.
    unsigned int  singular_quadrature_order;

    ///Size of the matrix = number of equations.
    unsigned int matrix_size;
    
    ///Linear system matrix.
    FullMatrix<double>  system_matrix;   
    ///Vector of BEM solution.
    Vector<double>      bem_solution;
    ///Vector of RHS
    Vector<double>      system_rhs;
    ///Vector of constant \f$ c(P) \f$, where \f$ P \f$ is a node of the boundary grid.
    Vector<double>      alpha;
    
};

#endif  //Bem_model_h