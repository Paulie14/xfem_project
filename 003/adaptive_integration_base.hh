#ifndef ADAPTIVE_INTEGRATION_BASE_H
#define ADAPTIVE_INTEGRATION_BASE_H

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_accessor.h>

#include "xfevalues.hh"
#include "mapping.hh"

//forward declarations
namespace dealii{
    template<int> class Function;
    template<int,int> class FE_Q;
}

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
    AdaptiveIntegrationBase(const dealii::DoFHandler<2>::active_cell_iterator &cell, 
                            const dealii::FE_Q<2,2> &fe,
                            XQuadratureBase* quadrature,
                            unsigned int m
                            );

    /// Sets the dirichlet and right hand side functors.
    void set_functors(dealii::Function<2>* dirichlet_function, 
                      dealii::Function<2>* rhs_function);
    
  protected: 
    
    ///Current cell to integrate
    const dealii::DoFHandler<2>::active_cell_iterator cell;
    
    ///Finite element of FEM
    const dealii::FE_Q<2,2>* fe;
    
    XQuadratureBase* xquad_;
    
        /// Index of aquifer on which we integrate.
    unsigned int m_;
    
    ///pointer to XFEM data belonging to the cell
    XDataCell* xdata;
    
    ///Pointer to function describing Dirichlet boundary condition.
    dealii::Function<2>* dirichlet_function;         
    ///Pointer to function describing RHS - sources.
    dealii::Function<2>* rhs_function;
};



/************************************ INLINE IMPLEMENTATION **********************************************/

inline void AdaptiveIntegrationBase::set_functors(Function< 2 >* dirichlet_function, 
                                                  Function< 2 >* rhs_function)
{
    this->dirichlet_function = dirichlet_function;
    this->rhs_function = rhs_function;
}


#endif  //ADAPTIVE_INTEGRATION_BASE_H