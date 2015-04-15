#include "adaptive_integration_base.hh"
// #include "system.hh"
// #include "data_cell.hh"

#include <deal.II/fe/fe_q.h>
using namespace dealii;

AdaptiveIntegrationBase::AdaptiveIntegrationBase(XDataCell* xdata, 
                                                 const DealFE_Q< 2 >& fe, 
                                                 XQuadratureBase* quadrature,
                                                 unsigned int m)
  : xdata_(xdata), fe_(&fe), xquad_(quadrature), m_(m),
    dirichlet_function_(nullptr),
    rhs_function_(nullptr)
{}


