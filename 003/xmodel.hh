#ifndef XModel_h
#define XModel_h

#include "model_base.hh"
#include "xfevalues.hh"


#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_accessor.h>
 
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

//block things
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <lac/block_matrix_array.h>




using namespace dealii;

// forward declarations
class Well;
class XDataCell;
class XQuadratureWell;
class GlobalSettingWriter;
namespace compare {
    class ExactBase;
}
namespace dealii{
    template<int,int> class PersistentTriangulation;
    template<int dim, int spacedim> class DoFHandler;
}

/** \mainpage
 * 
 * This is the practical part of the diploma thesis \f$ \textbf{Partition of unity methods for approximation of point water sources in porous media.} \f$ by Bc. Pavel Exner.
 * 
 * <b> Technical University of Liberec, Faculty of Mechatronics, Informatics and Interdisciplinary Studies. </b>
 *
 * Program: <a> Applied Sciences Engineering </a>, Branch: Science Engineering (Mathematical Modeling)
 * 
 * This program includes three experimental models of one aquifer flow problem with wells.
 * - model solved by FEM method with adaptively refined grid
 * - model solved by XFEM method 
 * - model solved by BEM method (is implemented but not working correctly)
 * 
 * We will solve system of equations
 * \f{equation} \label{eqn:laplace} 
 *    -T^m\Delta{}h^m=f^m  \qquad \textrm{na } \Theta^m \textrm{ pro } m=1,\ldots,M,
 * \f}
 * which is the Poisson equation derived from Darcy law and the equation of continuity. 
 * \f$ T^m \, [\textrm{m}^2\textrm{s}^{-1}] \f$ is scalar constant of transmisivity of the \f$ m \f$-th aquifer, 
 * \f$ f^m\, [\textrm{m}\textrm{s}^{-1}] \f$ is source density and \f$ h^m \f$ is pressure head in the \f$ m \f$-th aquifer.
 * 
 * and
 * \f{equation} \label{eqn:well_eqs_test}
      \int_{\partial{}B_w^m}\sigma_w^m \left(h^m - H_w^m\right)\bar{v}^m_w \, \mathrm{d}\mathbf{x}
      = c_w^{m+1}\left( H^m_w-H_w^{m+1}\right)\bar{v}^m_w - c^m_w\left( H^{m-1}_w-H^m_w \right)\bar{v}^m_w.
 * \f}
 *
 * where \f$ H^m_w \f$ is the pressure in th well in the level of \f$ m \f$-th aquifer, 
 * \f$ \partial{}B^m_w \f$ is the boundary of the \f$ w \f$-th well in the \f$ m \f$-th aquifer
 * and \f$ c^m_w \f$ is coeficient of conductivity of the \f$ w \f$-th well between \f$ (m-1)\f$-th and \f$ m \f$-th aquifer.
 * 
 * For detailed descrtiption (especially discrete formulation) see the text of diploma thesis.
 * 
 * To run the program, one need to have installed the library Deal II (version 7.2. was used while developing).
 */

  
///type of grid creation
    struct Well_computation 
    { 
      typedef enum {
        bc_newton,  //wells act like newton boundary condition   
        sources     //wells act like sources
      }Type;  
    };

/// class XModel
/**
 * @brief Model of an aquifer with wells using XFEM on non-adaptively refined grid.
 * 
 * More aquifers not implemented sofar.
 * Enrichment area is defined by \f$ r_{enr} \f$, which is set by @p set_enrichment_radius.
 * 
 */
class XModel : public ModelBase 
{
  public:
    /// Default constructor
    XModel ();
    
    /** Constructor
      * @param name is the name of the model
      * @param n_aquifers is the number of aquifers (not used)
      */
    XModel(const std::string &name, 
           const unsigned int &n_aquifers);
    
    /** Constructor
     * @param wells is the vector of wells
      * @param name is the name of the model
      * @param n_aquifers is the number of aquifers (not used)
      */
    XModel(const std::vector<Well*> &wells,
           const std::string &name="XFEM_Model", 
           const unsigned int &n_aquifers=1);
    
    /// Destructor
    virtual ~XModel ();
    
    /// @name Output
    //@{
    /** @brief Outputs solution on the computational mesh.
     * 
     * It still must compute enriched and unenriched part of solution at a point sum it up.
     * @param cycle is the number of a iteration, if the model is run in cycles
     */
    virtual void output_results (const unsigned int cycle=0) override;
    
    
    /** @brief Outputs solution on given mesh.
     * 
     * Computes solution on the mesh loaded from mesh file.
     * Creates the triangulation object and calls the other @p output_distributed_solution.
     * @param mesh_file is a GMSH file with mesh
     * @param flag_file is a file with refinement flags (is optional)
     * @param is_circle must be true if we load circle mesh with refinement flags 
     * @param cycle is the number of a iteration, if the model is run in cycles
     * @param m_aquifer number of aquifer which is meshed with the mesh loaded (not used)
     * * TODO: OBSOLETE - remove or inovate
     */
    virtual void output_distributed_solution (const std::string &mesh_file, 
                                              const std::string &flag_file, 
                                              bool is_circle=false,
                                              const unsigned int &cycle=0, 
                                              const unsigned int &m_aquifer=0) override;
    
    /** @brief Computes the solution on the given triangulation.
     * 
     * It can also deal with hanging nodes but only if they lie outside the enriched area.
     * 
     * @param tria is the output triangulation.
     * @param cycle is the number of a iteration, if the model is run in cycles
     * @param m_aquifer number of aquifer which is meshed with the mesh loaded (not used)
     * * TODO: OBSOLETE - remove or inovate
     */
    void output_distributed_solution (const Triangulation<2> &dist_tria,
                                      const unsigned int &cycle=0, 
                                      const unsigned int &m_aquifer=0);
    
    
    void compute_interpolated_exact(compare::ExactBase *exact_solution);
    
    /** Computes decomposed (enriched, unenriched and complete) solution at specified points.
     * It is called by @p output_distributed_solution with the nodes of the output triangulation.
     * @param points is the vector of points where we want to evaluate
     * TODO: OBSOLETE - remove or inovate
     */
    template<Enrichment_method::Type EnrType> 
    void compute_distributed_solution(const std::vector< Point< 2 > >& points, unsigned int m);
    //@}
    
    /// @name Getters
    //@{
    /** Returns constant reference to distributed solution.
     */
    const Vector< double > &get_solution() override;
    
    /** Returns constant reference to distributed solution.
     */
    const Vector< double > &get_distributed_solution() override;
    
    ///Returns reference to the computational triangulation.
    const Triangulation<2> &get_triangulation();
    
    /** Returns the number of degrees of freedom.
     * @return pair consisting of number of standard dofs and enriched dofs
     */
    std::pair<unsigned int, unsigned int> get_number_of_dofs();
    
    ///Returns reference to the output triangulation.
    const Triangulation<2> & get_output_triangulation();
    
    /// Returns pressure at the top of the well after computation.
    double well_pressure(unsigned int w);
    
    double well_band_width_ratio(void);
    //@}
    
    
    /// @name Setters
    //@{
    /// Sets the enrichment radius. Note that it can changed if it was chosen incorrectly.
    void set_enrichment_radius(double r_enr);
       
    /**Sets file path to a mesh file. 
     * Grid creation type @p grid_create is set to @p load.
     * @param coarse_mesh is the path to the GMSH file to be loaded
     * @param ref_flags is the path to the file with refinement flags
     */
    void set_computational_mesh (std::string coarse_mesh, std::string ref_flags = "");

    /// Sets the behavior of the well in the computation - boundary x source.
    void set_well_computation_type(Well_computation::Type well_computation);
    
    /// Sets the type of PUM method - the enrichment method.
    void set_enrichment_method (Enrichment_method::Type enrichment_method);
    
    /// Sets the adaptive integration refinement controlled by tolerance.
    /** It is switched off and the refinement controlled by geometry 
     * (distance from well center and well edge intersection) is used by default.
     * @param alpha_tolerance is 1e-2 by default.
     */
    void set_adaptive_refinement_by_error(double alpha_tolerance = 1e-2);
    
    
    void set_well_band_width_ratio(double band_ratio);
    //@}

    std::pair<double, double> integrate_difference(Vector<double>& diff_vector, 
                                                   compare::ExactBase * exact_solution, 
                                                   bool h1=false) override;

    /// @name Testing methods
    //@{
    /// Set the solution vector (dofs) accurately according to the exact solution and test XFEValues etc.
    void test_method(compare::ExactBase *exact_solution);
    
    /// Computes the H1 norm of [log - approx(log)] on elements on the band of enrichment radius.
    void test_enr_error();
    
    /// Tests adaptive integration - integrates characteristic function of a circle inside a square.
    double test_adaptive_integration(Function<2> *func, unsigned int level, unsigned int pol_degree=0);
    //@}
    
  protected:  
    virtual void make_grid () override;
    virtual void refine_grid() override;
    void setup_system () override;
    void assemble_system () override;
    void solve ();
   
    void assemble_dirichlet(unsigned int m) override;
    void setup_subsystem (unsigned int m);
    void assemble_subsystem (unsigned int m);
    void assemble_well_permeability_term(unsigned int m);
    void assemble_communication ();
    
    /// Procedures that are done in constructor.
    void constructor_init();
    
    // Print the list of Xdata
    void print_xdata();
    
    // Sets known dofs. For testing only. Does not work any more.
//     void assemble_reduce_known(unsigned int m);
    
    //Computes error in comparision to another solution
    //void compute_solution_error();

    /** @brief Function finds cells in which the centers of wells lies.
     * First, it finds the cell at which the center of the well lies and then
     * calls @p enrich_cell to recusively mark enriched cells and distribute 
     * degrees of freedom of the enrichment.
     */
    void find_enriched_cells(unsigned int m);
     
    /** @brief Recursive function to mark enriched cells and nodes.
     * Recusively marks enriched cells and distribute 
     * degrees of freedom of the enrichment.
     * Uses @p user_flags of the triangulation and tests only cells inside or on the edge
     * of the well. That means it will not go through all the cells.
     */
    void enrich_cell_blend (const DoFHandler<2>::active_cell_iterator cell, 
                            const unsigned int &well_index,
                            std::vector<unsigned int> &enriched_dof_indices,
                            std::vector<unsigned int> &enriched_weights,
                            unsigned int &n_global_enriched_dofs,
                            unsigned int m
                            );
   
    /** @brief Recursive function to mark enriched cells and nodes.
     * Recusively marks enriched cells and distribute 
     * degrees of freedom of the enrichment.
     * Uses @p user_flags of the triangulation and tests only cells inside or on the edge
     * of the well. That means it will not go through all the cells.
     */
    void enrich_cell (const DoFHandler<2>::active_cell_iterator cell, 
                      const unsigned int &well_index,
                      std::vector<unsigned int> &enriched_dof_indices,
                      unsigned int &n_global_enriched_dofs,
                      unsigned int m
                      );
    
    /** @brief Helpful function used when outputing the enriched shape function.
     * 
     * It returns in the vector @p cells all the cells, that are enriched by the given degree of freedom.
     * @param cells is the output vector of cells
     * @param dof_index is the index of the degree of freedom which we are testing
     */
    void find_dofs_enriched_cells(std::vector<DoFHandler<2>::active_cell_iterator> &cells, 
                                  const unsigned int &dof_index,
                                  unsigned int m
                                 );  
    
    /** Computes chosen shape function at specified points.
     * @param points is the vector of points where we want to evaluate
     * @param dof_index is the index of degree of freedom for which we want to compute the function
     * @param dof_func is a vector in which the values will be stored.
     * @param xfem is true then the xfem part of the function will be computed
     */
    void get_dof_func(const std::vector< Point< 2 > >& points, 
                      const unsigned int& dof_index,
                      Vector< double >& dof_func,
                      bool xfem = true
                     );
    
    void compute_well_quadratures();
    
    template<Enrichment_method::Type EnrType> 
    void prepare_shape_well_averiges(std::vector<std::map<unsigned int, double> > &shape_well_averiges,
                                     std::vector<XDataCell*> xdata);
    
    /// TODO:rename
    template<Enrichment_method::Type EnrType> 
    int recursive_output(double tolerance, 
                         PersistentTriangulation<2,2> &output_grid, 
                         DoFHandler<2> &temp_dof_handler, 
                         FE_Q<2> &temp_fe, 
                         const unsigned int iter,
                         unsigned int m
                        );
    
    template<Enrichment_method::Type EnrType>
    std::pair<double, double> integrate_difference(Vector<double>& diff_vector, 
                                                   compare::ExactBase * exact_solution);
    
    ///Type of enrichment method
    Enrichment_method::Type enrichment_method_;
    
    //Type of well compuation. How do we consider the well.
    Well_computation::Type well_computation_;
        
    std::string mesh_file;          ///< Path to output computational mesh. Should be removed.
    std::string coarse_grid_file;   ///< Path to computational coarse mesh.
    std::string ref_flags_file;     ///< Path to refinement flag file of the computational mesh.
    
    ///User defined enrichment radius.
    double rad_enr;
    
    ///Real enrichment radius used in computation.
    ///It may be different for each wells.
    std::vector<double> r_enr;
    
    std::vector<std::vector<XDataCell*> > xdata_;       ///< Vector of data for enriched cells.
    std::vector<std::vector<void *> > tria_pointers_;   ///< Vector of cell_pointers (XDataCell) of the tria.
    
    ///vector of global enrichment function values at nodes of the triangulation
    std::vector<std::vector<std::map<unsigned int, double> > > node_enrich_values;
    
    ///vector of global enrichment function values at nodes of the triangulation
    std::vector<std::map<unsigned int, double> > shape_well_averiges;
    
    /// Vector of quadratures in polar coordinates in vicinity of wells.
    std::vector<XQuadratureWell* > well_xquadratures_;
    
    unsigned int n_enriched_dofs_,   ///< Number of enriched degrees of freedom per aquifer.
                 n_standard_dofs_,   ///< Number of enriched degrees of freedom per aquifer.
                 n_dofs_;            ///< Number of enriched degrees of freedom per aquifer.
    
    ///2d triangulation of the aquifer
    PersistentTriangulation<2,2> *triangulation;
    ///coarse 2d triangulation of the aquifer, used by @p triangulation.
    Triangulation<2>            coarse_tria;
    
    //must be in this order, else a desctructor is needed to call dof_handler.clear();
    ///Finite 2d Element
    FE_Q<2>              fe;                    
    ///DofHandler for 2d triangulation
    DoFHandler<2>*      dof_handler;
    
    ///Gauss Quadrature for 2d finite element, 2-point (approximate 2d linear function), only on unenriched cells.
    QGauss<2>  quadrature_formula;
    ///Finite Element Value
    FEValues<2> fe_values;
    
    bool refine_by_error_;
    double alpha_tolerance_;
    
    ///takes care hanging nodes
    ConstraintMatrix hanging_node_constraints;
    ///is true if hanging nodes are present in the triangulation
    bool hanging_nodes;

    SparsityPattern               block_sp_pattern; ///< Shared sparsity pattern for a block.
    std::vector<SparsityPattern>  comm_sp_pattern;  ///< Shared sparsity patterns for a communication block.

    std::vector<SparseMatrix<double> >  block_matrix;      ///< Block diagonal aquifer matrices.
    std::vector<SparseMatrix<double> >  block_comm_matrix; ///< Communication matrices F_i.
    
    BlockMatrixArray<double> system_matrix_;    ///< System matrix.
    BlockVector<double> block_solution;         ///< Solution vector.
    BlockVector<double> block_system_rhs;       ///< Right hand side.
    
    Vector<double> dist_unenriched;     ///< Output vector - unenriched part of solution.
    Vector<double> dist_enriched;       ///< Output vector - enriched part of solution.
    Vector<double> dist_solution;       ///< Output vector - complete solution.
    
    PersistentTriangulation<2,2>* output_triangulation;   ///< Output triangulation (adaptively refined).
    
    
    /// Polar quadrature / edge rules switch
    static const bool use_polar_quadrature_;
    /// Width of the polar quadrature band around the well.
    double well_band_width_ratio_;
    
    /// Level of refinement for polar quadrature in the vicinity of a well
    static const unsigned int polar_refinement_level_;
    static const unsigned int well_band_gauss_degree_;
    static const unsigned int well_band_n_phi_;
    
    friend class GlobalSettingWriter;
};

#include "xmodel_impl.hh"


#endif  //XModel_h