#ifndef ADAPTIVE_INTEGRATION_BASE_H
#define ADAPTIVE_INTEGRATION_BASE_H

//forward declarations
namespace dealii{
    template<int> class Function;
    template<int,int> class FE_Q;
}
template<int dim,int spacedim=dim> using DealFE_Q = dealii::FE_Q<dim,spacedim>;

class Well;
class XDataCell;
class XQuadratureBase;


/** @brief A base class doing adaptive integration (in respect to the boundary of the well) on the cell.
 * 
 * Provides common functions and members to adaptive integration and unit cell refinement.
 */
class AdaptiveIntegrationBase
{
  public:
    /** @brief Constructor.
     * 
      * @param cell is cell iterator for the cell to be adaptively integrated.
      * @param fe is finite element used in FEM on this cell
      * @param mapping is mapping object that maps real cell to reference cell
      */ 
    AdaptiveIntegrationBase(XDataCell * xdata, 
                            const DealFE_Q<2> &fe,
                            XQuadratureBase* quadrature,
                            unsigned int m
                            );

    /// Sets the dirichlet and right hand side functors.
    void set_functors(dealii::Function<2>* dirichlet_function, 
                      dealii::Function<2>* rhs_function);
    
  protected: 
    ///pointer to XFEM data belonging to the cell
    XDataCell* xdata_;
    
    ///Finite element of FEM
    const DealFE_Q<2>* fe_;
    
    /// Pointer to quadrature base object.
    XQuadratureBase* xquad_;
    
        /// Index of aquifer on which we integrate.
    unsigned int m_;
    
    ///Pointer to function describing Dirichlet boundary condition.
    dealii::Function<2>* dirichlet_function_;         
    ///Pointer to function describing RHS - sources.
    dealii::Function<2>* rhs_function_;
};



/************************************ INLINE IMPLEMENTATION **********************************************/

inline void AdaptiveIntegrationBase::set_functors(dealii::Function< 2 >* dirichlet_function, 
                                                  dealii::Function< 2 >* rhs_function)
{
    dirichlet_function_ = dirichlet_function;
    rhs_function_ = rhs_function;
}


#endif  //ADAPTIVE_INTEGRATION_BASE_H