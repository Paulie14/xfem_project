#ifndef XModel_h
#define XModel_h

#include <deal.II/grid/tria.h>
#include <deal.II/grid/persistent_tria.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_accessor.h>
 
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/vector.h>

//block things
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_matrix.h>
#include <deal.II/lac/block_vector.h>

#include "model_base.hh"
#include "xfevalues.hh"

using namespace dealii;

class Well;
class XDataCell;

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
class XModel : public Model_base 
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
     */
    void output_distributed_solution (const Triangulation<2> &dist_tria,
                                      const unsigned int &cycle=0, 
                                      const unsigned int &m_aquifer=0);
    
    
    /** Computes decomposed (enriched, unenriched and complete) solution at specified points.
     * It is called by @p output_distributed_solution with the nodes of the output triangulation.
     * @param points is the vector of points where we want to evaluate
     */
    void compute_distributed_solution(const std::vector< Point< 2 > >& points);
    //@}
    
    /// @name Getters
    //@{
    /** Returns constant reference to distributed solution.
     */
    const Vector< double > &get_solution() override;
    
    /** Returns constant reference to distributed solution.
     */
    const Vector< double > &get_distributed_solution() override;
    
    ///Returns reference to the computational grid
    inline const Triangulation<2> &get_triangulation()
    { return *triangulation; }
    
    ///Returns the total number of degrees of freedom (both enriched and unenriched)
    std::pair<unsigned int, unsigned int> get_number_of_dofs()
    { //std::pair<unsigned int, unsigned int> pair(n_enriched_dofs, dof_handler->n_dofs()); 
      return std::make_pair(dof_handler->n_dofs(), n_enriched_dofs);
    }
    
    inline const Triangulation<2> & get_output_triangulation()
    { return *output_triangulation;}
    
    //unsigned int get_number_of_dofs()
    //{ return n_enriched_dofs + dof_handler->n_dofs(); }
    //@}
    
    
    /// @name Setters
    //@{
    inline void set_enrichment_radius(double r_enr)
    { this->rad_enr = r_enr; }
    
    inline void set_output_features(bool decomposed = true, bool shape_functions = false)
    { this->out_decomposed = decomposed;
      this->out_shape_functions = shape_functions;
    }
   
    /**Sets file path to a mesh file. 
     * Grid creation type @p grid_create is set to @p load.
     * @param coarse_mesh is the path to the GMSH file to be loaded
     * @param ref_flags is the path to the file with refinement flags
     */
    inline void set_computational_mesh (std::string coarse_mesh, std::string ref_flags = "")
    { this->coarse_grid_file = coarse_mesh;
      this->ref_flags_file = ref_flags;
      grid_create = load;
    }
    
    inline void set_well_computation_type(Well_computation::Type well_computation)
    {
      well_computation_ = well_computation;
    }
    
    inline void set_enrichment_method (Enrichment_method::Type enrichment_method)
    { enrichment_method_ = enrichment_method;}
    //@}
    
    //returns vector of support points 
    //void get_support_points (std::vector<Point<2> > &support_points);
    //output results from another model but computed on the model mesh
    //void output_foreign_results(const unsigned int cycle,const Vector<double> &foreign_solution);


  protected:
    virtual void make_grid () override;
    virtual void refine_grid() override;
    void setup_system () override;
    void assemble_system () override;
    void solve ();
   
    
    //Computes error in comparision to another solution
    //void compute_solution_error();

    /** @brief Function finds cells in which the centers of wells lies.
     * First, it finds the cell at which the center of the well lies and then
     * calls @p enrich_cell to recusively mark enriched cells and distribute 
     * degrees of freedom of the enrichment.
     */
    void find_enriched_cells();
     
    /** @brief Recursive function to mark enriched cells and nodes.
     * Recusively marks enriched cells and distribute 
     * degrees of freedom of the enrichment.
     * Uses @p user_flags of the triangulation and tests only cells inside or on the edge
     * of the well. That means it will not go through all the cells.
     */
    void enrich_cell (const DoFHandler<2>::active_cell_iterator cell, 
                      const unsigned int &well_index,
                      std::vector<unsigned int> &enriched_dof_indices,
                      std::vector<unsigned int> &enriched_weights,
                      unsigned int &n_global_enriched_dofs
                      );
   
    /** @brief Recursive function to mark enriched cells and nodes.
     * Recusively marks enriched cells and distribute 
     * degrees of freedom of the enrichment.
     * Uses @p user_flags of the triangulation and tests only cells inside or on the edge
     * of the well. That means it will not go through all the cells.
     */
    void enrich_cell_sgfem (const DoFHandler<2>::active_cell_iterator cell, 
                      const unsigned int &well_index,
                      std::vector<unsigned int> &enriched_dof_indices,
                      unsigned int &n_global_enriched_dofs
                      );
    
    /** @brief Helpful function used when outputing the enriched shape function.
     * 
     * It returns in the vector @p cells all the cells, that are enriched by the given degree of freedom.
     * @param cells is the output vector of cells
     * @param dof_index is the index of the degree of freedom which we are testing
     */
    void find_dofs_enriched_cells(std::vector<DoFHandler<2>::active_cell_iterator> &cells, 
                                  const unsigned int &dof_index);
    
    
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
    
    template<Enrichment_method::Type> 
    int recursive_output(double tolerance, PersistentTriangulation<2> &output_grid, DoFHandler<2> &temp_dof_handler, FE_Q<2> &temp_fe, const unsigned int cycle);
    
    ///Type of enrichment method
    Enrichment_method::Type enrichment_method_;
    
    //Type of well compuation. How do we consider the well.
    Well_computation::Type well_computation_;
        
    /** path to computational mesh
     * temporary, should be removed (only for output)
     */
    std::string mesh_file;
    
    ///path to computational mesh
    std::string coarse_grid_file;
    ///path to refinement flag file
    std::string ref_flags_file;
    
    ///User defined enrichment radius.
    double rad_enr;
    
    ///Real enrichment radius used in computation.
    ///It may be different for each wells.
    std::vector<double> r_enr;
    
    ///vector of data for enriched cells
    std::vector<XDataCell*> xdata;
    
    ///vector of global enrichment function values at nodes of the triangulation
    std::vector<std::map<unsigned int, double> > node_enrich_values;
    
    ///Number of enriched degrees of freedom
    unsigned int n_enriched_dofs;
    
    //number of quadrature points on well
    //const unsigned int n_well_qpoints;
    
    ///2d triangulation of the aquifer
    PersistentTriangulation<2> *triangulation;
    ///coarse 2d triangulation of the aquifer, used by @p triangulation.
    Triangulation<2>            coarse_tria;
    
    //must be in this order, else a desctructor is needed to call dof_handler.clear();
    ///Finite 2d Element
    FE_Q<2>              fe;                    
    ///DofHandler for 2d triangulation
    DoFHandler<2>        *dof_handler;
    
    ///Gauss Quadrature for 2d finite element, 2-point (approximate 2d linear function), only on unenriched cells.
    QGauss<2>  quadrature_formula;
    ///Finite Element Value
    FEValues<2> fe_values;
    
    
    ///takes care hanging nodes
    ConstraintMatrix hanging_node_constraints;
    ///is true if hanging nodes are present in the triangulation
    bool hanging_nodes;

    ///Sparsity pattern of the system matrix.
    BlockSparsityPattern        block_sp_pattern;
    ///System Matrix
    BlockSparseMatrix<double>   block_matrix;   
    
    ///Solution vector (unenriched, enriched and well degrees of freedom)
    BlockVector<double> block_solution;
    ///Right hand side
    BlockVector<double> block_system_rhs;
    
    //Vector<double>      solution_error;
    //Vector<double>      solution_exact;
    
    ///Output vector - unenriched part of solution
    Vector<double> dist_unenriched;
    ///Output vector - enriched part of solution
    Vector<double> dist_enriched;
    ///Output vector - complete solution
    Vector<double> dist_solution;
    
    //output
    bool out_decomposed;
    bool out_shape_functions;
    PersistentTriangulation<2>* output_triangulation;
};

#include "xmodel_impl.hh"


#endif  //XModel_h