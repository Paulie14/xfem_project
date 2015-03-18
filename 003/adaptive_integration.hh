#ifndef ADAPTIVE_INTEGRATION_H
#define ADAPTIVE_INTEGRATION_H

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/quadrature_lib.h>

#ifdef DEBUG
 // #define DECOMPOSED_CELL_MATRIX
#endif

//#define SOURCES
#define BC_NEWTON
  
#include "mapping.hh"
#include "xfevalues.hh"


//forward declarations
namespace dealii{
    template<int> class Function;
    template<int,int> class FE_Q;
}

class Well;
class XDataCell;

/** @brief Class representing squares of adaptive refinement of the reference cell.
 * 
 * This class owns vertices of the squares, mapping procedures between square and unit cell.
 */
class Square
{
public: 
    ///@brief Constructor.
    ///@param p1 is down left vertex of the square
    ///@param p2 is up right vertex of the square
    Square(const Point<2> &p1, const Point<2> &p2);
 
    ///@name Getters
    //@{
        double real_diameter() const;               ///< Returns diameter in real coordinates.
        double unit_diameter() const;               ///< Returns diameter in unit cell coordinates.
        dealii::Point<2> real_vertex(unsigned int i) const; ///< Returns @p i vertex in real coordinates.
        dealii::Point<2> vertex(unsigned int i) const;      ///< Returns @p i vertex in unit cell coordinates.
        dealii::Quadrature<2> const* quadrature() const;    ///< Returns square quadrature.
    //@}
    
    /// Transforms the square into the real coordinates.
    void transform_to_real_space(const dealii::DoFHandler< 2  >::active_cell_iterator& cell,
                                 const dealii::Mapping<2> &mapping);
    
  /** Vertices of the square.
    *
    * Numbering of the squares:
    * (is different from DealII)
    *       2
    *   3-------2
    *   |       |
    * 3 |       | 1
    *   |       |
    *   0-------1
    *       0
    */
    dealii::Point<2> vertices[4];
  
    ///Object mappping data between the adaptively created square and unit cell
    MyMapping mapping;
  
    ///Refine flag is set true, if this square should be refined during next refinement run.
    bool refine_flag;
  
    /// Flag is true if the square has already been processed.
    bool processed;
  
    ///Pointer to Gauss quadrature, that owns the quadrature points and their weights.
    dealii::QGauss<2> const *gauss;
  
private:
    dealii::Point<2> real_vertices_[4];
    /// Length of diagonal in real space.
    double real_diameter_;
    /// Length of diagonal in real space.
    double unit_diameter_;
    /// Flag is true if the @p transform_to_real_space method was called.
    bool transformed_to_real_;
};


/** @brief class doing adaptive integration (in respect to the boundary of the well) on the cell.
 * 
 *  First, it tests if the cell include the well, then does the refinemnt according to the criterion.
 *  Then computes local matrix.
 */
class Adaptive_integration
{
  public:
    /** @brief Constructor.
     * 
      * @param cell is cell iterator for the cell to be adaptively integrated.
      * @param fe is finite element used in FEM on this cell
      * @param mapping is mapping object that maps real cell to reference cell
      */ 
    Adaptive_integration(const dealii::DoFHandler<2>::active_cell_iterator &cell, 
                         const dealii::FE_Q<2,2> &fe,
                         const dealii::Mapping<2>& mapping,
                         unsigned int m
                        );
    
    /// Getter for current level of refinement
    unsigned int level();

    /// Sets the dirichlet and right hand side functors.
    void set_functors(dealii::Function<2>* dirichlet_function, 
                      dealii::Function<2>* rhs_function);
    
    /// @brief Refinement along the well edge.
    /** If the square is crossed by the well edge
      * it will be refined.
      */
    bool refine_edge();
    
    /// @brief Refinement controlled by chosen tolerance.
    bool refine_error(double alpha_tolerance = 1e-2);
    
    /** @brief Integrates over all squares and their quadrature points.
      * @tparam EnrType is the type of enrichment method (XFEM-shifted, SGFEM sofar)
      * @param cell_matrix is a local matrix of the cell
      * @param cell_rhs is a local rhs of the local matrix
      * @param local_dof_indices is vector of dof indices (both unenriched and enriched)
      * @param transmisivity is transmisivity defined on the cell for the Laplace member of the equation
      */        
    template<Enrichment_method::Type EnrType> 
    void integrate( dealii::FullMatrix<double> &cell_matrix, 
                    dealii::Vector<double> &cell_rhs,
                    std::vector<unsigned int> &local_dof_indices,
                    const double &transmisivity
                    );
    
    /** @brief Integrates over all squares and their quadrature points the L2 norm of difference to exact solution.
      * @tparam EnrType is the type of enrichment method (XFEM-shifted, SGFEM sofar)
      * @param solution is the vector of computed degrees of freedom, i.e. solution
      * @param exact_solution is the functor representing the exact solution
      */   
    template<Enrichment_method::Type EnrType> 
    double integrate_l2_diff(const dealii::Vector<double> &solution, 
                             const dealii::Function<2> &exact_solution);
    
//     /** OBSOLETE First version of XFEM (without shift).
//      * Does everything inside - no XFEValues.
//      */
//     void integrate_xfem( FullMatrix<double> &cell_matrix, 
//                          Vector<double> &cell_rhs,
//                          std::vector<unsigned int> &local_dof_indices,
//                          const double &transmisivity
//                        );
//     
//     /** OBSOLETE XFEM with shift.
//      * Adds shift to @p integrate_xfem().
//      * Does everything inside - no XFEValues.
//      */
//     void integrate_xfem_shift( FullMatrix<double> &cell_matrix, 
//                                Vector<double> &cell_rhs,
//                                std::vector<unsigned int> &local_dof_indices,
//                                const double &transmisivity
//                              );
//     
//     /** OBSOLETE XFEM with shift.
//      * Uses XFEValues for the first time, with quadrature points precomputed.
//      * Used as scheme for reimplementation in template method.
//      */
//     void integrate_xfem_shift2( FullMatrix<double> &cell_matrix, 
//                                Vector<double> &cell_rhs,
//                                std::vector<unsigned int> &local_dof_indices,
//                                const double &transmisivity
//                              );
//     
//     /** OBSOLETE SGFEM.
//      * First version of SGFEM method.
//      * Uses XFEValues for the first time, but not for gradients.
//      */
//     void integrate_sgfem( FullMatrix<double> &cell_matrix, 
//                           Vector<double> &cell_rhs,
//                           std::vector<unsigned int> &local_dof_indices,
//                           const double &transmisivity
//                         );
//     
//     /** OBSOLETE SGFEM.
//      * First version of SGFEM method.
//      * Does everything inside - no XFEValues.
//      */
//     void integrate_sgfem2( FullMatrix<double> &cell_matrix, 
//                           Vector<double> &cell_rhs,
//                           std::vector<unsigned int> &local_dof_indices,
//                           const double &transmisivity
//                         );
//     
//     /** OBSOLETE SGFEM.
//      * Third version of SGFEM method.
//      * Uses XFEValues with precomputed quadrature points.
//      * Used as scheme for reimplementation in template method.
//      */
//     void integrate_sgfem3( FullMatrix<double> &cell_matrix, 
//                           Vector<double> &cell_rhs,
//                           std::vector<unsigned int> &local_dof_indices,
//                           const double &transmisivity
//                         );
    

    
    /** @brief Calls gnuplot to create image of refined element.
     * 
      * Also can save the gnuplot script to file.
      * @param output_dir is the directory for output_dir
      * @param real is true then the element is printed in real coordinates
      * @param show is true then the gnuplot utility is started and plots the refinement on the screen
      */ 
    void gnuplot_refinement(const std::string &output_dir, bool real=true, bool show=false);
    
    /// @name Test methods
    //@{
        /// Test - integrates the @p func using adaptive integration (used for circle characteristic function).
        double test_integration(dealii::Function<2>* func); 
    
        /// Test - integrates the @p func using adaptive integration on different levels (used for \f$ 1/r^2) \f$.
        std::pair<double,double> test_integration_2(dealii::Function<2>* func, unsigned int diff_levels);
    //@}
    
  private: 
      
    ///Does the actual refinement of the squares according to the flags.
    /// @param n_squares_to_refine is number of squares to be refined
    void refine(unsigned int n_squares_to_refine);
    
    static bool remove_square_cond(Square sq) {return sq.refine_flag;}
    
    /// Gathers the quadrature points and their weigths from squares into a single vector.
    void gather_w_points();
    
    ///@name Refinement criteria.
    //@{
        /// Returns true if criterion is satisfied.
        /** Criterion: square diameter > C * (minimal distance of a node from well edge)
        */
        bool refine_criterion_a(Square &square, Well &well);
        
        /// Returns number of nodes of @p square inside the @p well.
        unsigned int refine_criterion_nodes_in_well(Square &square, Well &well);
        
        /// Computes the alpha criterion for different n (quadrature order) and returns the quad. order
        unsigned int refine_criterion_alpha(double r_min);
        
        bool refine_criterion_h(Square &square, Well &well, double criterion_rhs);
        
        /// Computes r_min
        double compute_r_min(Square &square, unsigned int w);
    //@}
    
    ///Current cell to integrate
    const dealii::DoFHandler<2>::active_cell_iterator cell;
    
    ///Finite element of FEM
    const dealii::FE_Q<2,2>  *fe;
    
    ///mapping from real cell to unit cell
    const dealii::Mapping<2> *mapping;
    
        /// Index of aquifer on which we integrate.
    unsigned int m_;
    
    ///pointer to XFEM data belonging to the cell
    XDataCell *xdata;
    
    ///TODO: Use only DealII mapping for cell_mapping
    ///mapping data of the cell to unit cell
    MyMapping cell_mapping;
    
    ///Vector of refined squares.
    /// square[i][j] -> i-th square and its j-th vertex
    std::vector<Square> squares;
    
    std::vector<dealii::Point<2> > q_points_all;
    std::vector<double> jxw_all;
    
    ///Pointer to function describing Dirichlet boundary condition.
    dealii::Function<2> *dirichlet_function;         
    ///Pointer to function describing RHS - sources.
    dealii::Function<2> *rhs_function;
  
    ///Level of current refinement.
    unsigned int level_;
    
    /// Tolerance for refine_error method.
    double alpha_tolerance_;
    
    
    ///TODO: Get rid of these
    ///helpful temporary data
    ///mapped well centers to unit cell
    std::vector<dealii::Point<2> > m_well_center;
    
    ///mapped well radius to unit cell
    std::vector<double > m_well_radius;
    
    // Square refinement criteria constant on the cells without well inside.
    static const double square_refinement_criteria_factor_;
    
    //alpha in apriori adaptive criterion
    static const std::vector<double> alpha_;
    
    /// Empiric constants for refine_error method.
    static const double c_empiric_, p_empiric_;
    
    /// Vector of Gauss quadrature of different order.
    static const std::vector<dealii::QGauss<2> > quadratures_;
};


#include "adaptive_integration_impl.hh"
#endif  //ADAPTIVE_INTEGRATION_H


