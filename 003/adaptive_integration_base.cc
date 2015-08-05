#include "adaptive_integration_base.hh"
// #include "system.hh"
#include "data_cell.hh"
// #include "well.hh"

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

// bool AdaptiveIntegrationBase::is_cell_in_well(unsigned int& w)
// {
//     DoFHandler<2>::active_cell_iterator cell = xdata_->get_cell();
//     
//     for(unsigned int ww = 0; ww < xdata_->n_wells(); ww++)
//     {
//         Well * well = xdata_->get_well(ww);
//         
//         bool cell_inside = true;
//         for(unsigned int i=0; i < GeometryInfo<2>::vertices_per_cell; i++)
//         {
//             cell_inside &= well->points_inside(cell->vertex(i)); //if not then it is outside and set false
//         }
//         
//         if(cell_inside) 
//         {
//             w = ww;
//             return true;
//         }
//     }
//     
//     w = -1; 
//     return false;
// }


