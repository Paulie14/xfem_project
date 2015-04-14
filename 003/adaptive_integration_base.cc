#include "adaptive_integration_base.hh"
#include "system.hh"
#include "data_cell.hh"
#include "mapping.hh"
#include "xquadrature_base.hh"

#include <iostream>

#include <deal.II/base/point.h>

AdaptiveIntegrationBase::AdaptiveIntegrationBase(const dealii::DoFHandler< 2  >::active_cell_iterator& cell,
                                                 const FE_Q< 2, 2 >& fe, 
                                                 XQuadratureBase* quadrature, 
                                                 unsigned int m)
  : cell(cell), fe(&fe), xquad_(quadrature), m_(m),
    dirichlet_function(nullptr),
    rhs_function(nullptr)
{
    //TODO: move to xmodel.cc a give it through constructor
      MASSERT(cell->user_pointer() != NULL, "NULL user_pointer in the cell"); 
      //A *a=static_cast<A*>(cell->user_pointer()); //from DEALII (TriaAccessor)
      xdata = static_cast<XDataCell*>( cell->user_pointer() );
}


