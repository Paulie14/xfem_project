#ifndef ADAPTIVE_INTEGRATION_H
#define ADAPTIVE_INTEGRATION_H

#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/function.h>

#ifdef DEBUG
 // #define DECOMPOSED_CELL_MATRIX
#endif

//#define SOURCES
#define BC_NEWTON
  
#include "mapping.hh"
#include "xfevalues.hh"

using namespace dealii;

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
        inline double real_diameter() const
        { return real_diameter_; }
        
        inline double unit_diameter() const
        { return unit_diameter_; }
        
        inline Point<2> real_vertex(unsigned int i) const
        { return real_vertices_[i]; }
        
        inline Point<2> vertex(unsigned int i) const
        { return vertices[i]; }
        
        inline Quadrature<2> const* quadrature() const
        { return gauss; }
    //@}
    
    void transform_to_real_space(const DoFHandler< 2  >::active_cell_iterator& cell,
                                 const Mapping<2> &mapping);
    
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
    Point<2> vertices[4];
  
    ///Object mappping data between the adaptively created square and unit cell
    MyMapping mapping;
  
    ///Refine flag is set true, if this square should be refined during next refinement run.
    bool refine_flag;
  
    /// Flag is true if the square has already been processed.
    bool processed;
  
    ///Pointer to Gauss quadrature, that owns the quadrature points and their weights.
    QGauss<2> const *gauss;
  
private:
    Point<2> real_vertices_[4];
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
    Adaptive_integration(const DoFHandler<2>::active_cell_iterator &cell, 
                         const FE_Q<2> &fe,
                         const Mapping<2>& mapping,
                         unsigned int m
                        );
    
    ///Getter for current level of refinement
    inline unsigned int get_level()
    {return level;}
    
    inline void set_functors(Function<2>* dirichlet_function, Function<2>* rhs_function)
    { this->dirichlet_function = dirichlet_function;
      this->rhs_function = rhs_function;
    }
    
    /// @brief Refinement along the well edge.
    /** If the square is crossed by the well edge
      * it will be refined.
      */
    bool refine_edge();
    
    /// @brief Refinement according to the error at each square.
    /** TODO: suggest error computation and comparing
     *  Sofar not implemented
      */ 
    bool refine_error(double alpha_tolerance = 1e-2);
    
    /** @brief Integrates over all squares and their quadrature points.
      * @tparam EnrType is the type of enrichment method (XFEM-shifted, SGFEM sofar)
      * @param cell_matrix is a local matrix of the cell
      * @param cell_rhs is a local rhs of the local matrix
      * @param local_dof_indices is vector of dof indices (both unenriched and enriched)
      * @param transmisivity is transmisivity defined on the cell for the Laplace member of the equation
      */        
    template<Enrichment_method::Type EnrType> 
    void integrate( FullMatrix<double> &cell_matrix, 
                    Vector<double> &cell_rhs,
                    std::vector<unsigned int> &local_dof_indices,
                    const double &transmisivity
                    );
    
    /** @brief Integrates over all squares and their quadrature points the L2 norm of difference to exact solution.
      * @tparam EnrType is the type of enrichment method (XFEM-shifted, SGFEM sofar)
      * @param solution is the vector of computed degrees of freedom, i.e. solution
      * @param exact_solution is the functor representing the exact solution
      */   
    template<Enrichment_method::Type EnrType> 
    double integrate_l2_diff(const Vector<double> &solution, const Function<2> &exact_solution);
    
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
    
    
    double test_integration(Function<2>* func); 
    
    std::pair<double,double> test_integration_2(Function<2>* func, unsigned int diff_levels);
    
  private: 
    ///Current cell to integrate
    const DoFHandler<2>::active_cell_iterator cell;
    
    ///Finite element of FEM
    const FE_Q<2>  *fe;
    
    ///mapping from real cell to unit cell
    const Mapping<2> *mapping;
    
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
    
    std::vector<Point<2> > q_points_all;
    std::vector<double> jxw_all;
    
    ///Pointer to function describing Dirichlet boundary condition.
    Function<2> *dirichlet_function;         
    ///Pointer to function describing RHS - sources.
    Function<2> *rhs_function;
  
    ///Level of current refinement.
    unsigned int level;
    
    double alpha_tolerance_;
    
    ///Does the actual refinement of the squares according to the flags.
    /// @param n_squares_to_refine is number of squares to be refined
    void refine(unsigned int n_squares_to_refine);
    
    inline static bool remove_square_cond(Square sq) {return sq.refine_flag;}
    
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
    
    void gather_w_points();
  
    
    ///TODO: Get rid of these
    ///helpful temporary data
    ///mapped well centers to unit cell
    std::vector<Point<2> > m_well_center;
    
    ///mapped well radius to unit cell
    std::vector<double > m_well_radius;
    
    // Square refinement criteria constant on the cells without well inside.
    static const double square_refinement_criteria_factor_;
    
    //alpha in apriori adaptive criterion
    static const std::vector<double> alpha_;
    
    static const double c_empiric_, p_empiric_;
    
//         /// 1 point Gauss quadrature with dim=2
//     static const QGauss<2> empty_quadrature;
//     /// 1 point Gauss quadrature with dim=2
//     static const QGauss<2> gauss_1;
//     /// 3 point Gauss quadrature with dim=2
//     static const QGauss<2> gauss_2;
//     /// 3 point Gauss quadrature with dim=2
//     static const QGauss<2> gauss_3;
//     /// 4 point Gauss quadrature with dim=2
//     static const QGauss<2> gauss_4;
    
    static const std::vector<QGauss<2> > quadratures_;
};


#include "adaptive_integration_impl.hh"
#endif  //ADAPTIVE_INTEGRATION_H


