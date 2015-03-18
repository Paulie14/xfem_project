
#ifndef XFEValues_h
#define XFEValues_h

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_values.h>
#include "system.hh"

namespace dealii{
    template<int,int> class FiniteElement;
}

class XDataCell;

/** Enumerates methods of enrichment.
   */
struct Enrichment_method
{
  typedef enum {xfem,        ///< standard XFEM
                xfem_ramp,   ///< XFEM method with ramp function only
                xfem_shift,  ///< XFEM method with ramp funciton and shift
                sgfem        ///< SGFEM method
  } Type;
};

/** Extended FEValues class for XFEM/SGFEM with point sources.
 * It provides computation of the enrichment shape functions.
 */
template<Enrichment_method::Type> 
class XFEValues : public dealii::FEValues< 2 >
{
public:
  XFEValues(const dealii::FiniteElement<2> &fe, const dealii::Quadrature<2> &quadrature, const dealii::UpdateFlags update_flags) 
    : dealii::FEValues<2>(fe, quadrature, update_flags),
      xdata_(nullptr)
  {
    n_vertices_ = dealii::GeometryInfo<2>::vertices_per_cell;
  }
  
  using dealii::FEValues<2>::reinit;
  
  /** Reinits the values for the enriched cell (node values etc.)
   * Note that it calls the reinit function of the Deal II FEValues.
   */
  void reinit(XDataCell* xdata);

  /** Returns the value of the enrichment test function at a quadrature point (according to used quadrature).
   */
  double enrichment_value(const unsigned int function_no, const unsigned int w, const unsigned int q);
  
  /** Returns the value of the enrichment test function at given point.
   */
  double enrichment_value(const unsigned int function_no, const unsigned int w, const dealii::Point<2> p);
  
  /** Returns the value of the gradient of the enrichment test function at quadrature point (according to used quadrature).
   */
  dealii::Tensor<1,2> enrichment_grad(const unsigned int function_no, const unsigned int w, const unsigned int q);
  
  /** NOT WORKING. Returns the value of the gradient of the enrichment test function at given point.
   */
  dealii::Tensor<1,2> enrichment_grad(const unsigned int function_no, const unsigned int w, const dealii::Point<2> p);
  
private:
  /// Vector of values of the enrichment test function at quadrature points.
  std::vector<std::vector<double> > q_enrich_values_;
  
  /// Vector of the ramp function values (XFEM) at quadrature points.
  std::vector<std::vector<double> > q_ramp_values_;
  
  /// Function that precomputes XFEM ramp function or SGFEM interpolation at quadrature points.
  void prepare();
  
  XDataCell* xdata_;                                    ///< Pointer to enrichment data.
  dealii::DoFHandler<2>::active_cell_iterator cell_;    ///< Cell iterator.
  unsigned int n_wells_;                                ///< Number of wells affecting the current cell.
  unsigned int n_vertices_;                             ///< Number of vertices per current cell.
};





/*******************************************     IMPLEMENTATION                   ***************************/

#include "well.hh"
#include "data_cell.hh"

template<Enrichment_method::Type T> 
void XFEValues<T>::reinit(XDataCell* xdata)
{
  xdata_ = xdata;
  n_wells_ = xdata_->n_wells();
  cell_ = xdata_->get_cell();
  this->reinit(cell_);
  
  
  if(dealii::update_quadrature_points & this->get_update_flags())
  {
    if(dealii::update_values & this->get_update_flags())
    {
      q_enrich_values_.resize(n_wells_);
  
      for(unsigned int w=0; w < n_wells_; w++)
      {
        q_enrich_values_[w].resize(this->n_quadrature_points);

        for(unsigned int q=0; q < this->n_quadrature_points; q++)
          q_enrich_values_[w][q] = xdata_->get_well(w)->global_enrich_value(this->quadrature_point(q)); //returns quadrature_point in real coordinates
      }
      prepare();
    }
  }
}



#endif // XFEValues