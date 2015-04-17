#ifndef ADAPTIVE_INTEGRATION_H
#define ADAPTIVE_INTEGRATION_H

#include <deal.II/base/function.h>

#ifdef DEBUG
 // #define DECOMPOSED_CELL_MATRIX
#endif

//#define SOURCES
#define BC_NEWTON
 
#include "adaptive_integration_base.hh"
#include "xfevalues.hh"


//forward declarations
namespace dealii{
//     template<int> class Function;
    template<int,int> class FE_Q;
}
template<int dim,int spacedim=dim> using DealFE_Q = dealii::FE_Q<dim,spacedim>;

// class Well;
class XQuadratureBase;


/** @brief class doing adaptive integration (in respect to the boundary of the well) on the cell.
 * 
 *  First, it tests if the cell include the well, then does the refinemnt according to the criterion.
 *  Then computes local matrix.
 */
class Adaptive_integration : public AdaptiveIntegrationBase
{
  public:
    /** @brief Constructor.
     * 
      * @param cell is cell iterator for the cell to be adaptively integrated.
      * @param fe is finite element used in FEM on this cell
      */ 
    Adaptive_integration(XDataCell * xdata, 
                         const DealFE_Q<2> &fe,
                         XQuadratureBase * quadrature,
                         unsigned int m
                        );
    
    
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
    
    /// @name Test methods
    //@{
        /// Test - integrates the @p func using adaptive integration (used for circle characteristic function).
        double test_integration(dealii::Function<2>* func); 
    
//         /// Test - integrates the @p func using adaptive integration on different levels (used for \f$ 1/r^2) \f$.
//         std::pair<double,double> test_integration_2(dealii::Function<2>* func, unsigned int diff_levels);
    //@}
    
  private: 

};





class SmoothStep : public dealii::Function<2>
{
public:
    
    SmoothStep(Well* well, double band_width);

    double value (const dealii::Point<2>   &p,
                  const unsigned int  component = 0) const override;
    double value (const double & r) const;
                  
private:
    Well * well_;
    double band_width_;
    double coefs_[6];
    unsigned int size_;
};

/** @brief class doing adaptive integration (in respect to the boundary of the well) on the cell.
 * 
 *  First, it tests if the cell include the well, then does the refinemnt according to the criterion.
 *  Then computes local matrix.
 */
class AdaptiveIntegrationPolar : public AdaptiveIntegrationBase
{
  public:
    /** @brief Constructor.
     * 
      * @param cell is cell iterator for the cell to be adaptively integrated.
      * @param fe is finite element used in FEM on this cell
      * @param mapping is mapping object that maps real cell to reference cell
      */ 
    AdaptiveIntegrationPolar(XDataCell* xdata, 
                             const DealFE_Q< 2 >& fe, 
                             XQuadratureBase* quadrature, 
                             std::vector< XQuadratureWell* > polar_xquads, 
                             unsigned int m
                            );

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
    
    static unsigned int n_point_check;
    
  private: 
    std::vector<XQuadratureWell*> polar_xquads_;
    

};

#include "adaptive_integration_impl.hh"
#endif  //ADAPTIVE_INTEGRATION_H


