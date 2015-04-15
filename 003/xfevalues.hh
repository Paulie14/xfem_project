
#ifndef XFEValues_h
#define XFEValues_h

#include <deal.II/fe/fe_values.h>

//due to implementation:
#include "system.hh"
#include "well.hh"
#include "data_cell.hh"
#include "xquadrature_base.hh"

// forward declarations
// class XDataCell;
// class XQuadratureBase;

/** Enumerates methods of enrichment.
   */
struct Enrichment_method
{
  typedef enum {xfem,        ///< standard XFEM
                xfem_ramp,   ///< XFEM method with ramp function only
                xfem_shift,  ///< XFEM method with ramp function and shift
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
  XFEValues(const dealii::FiniteElement<2> &fe, dealii::Quadrature<2> &quadrature, const dealii::UpdateFlags update_flags) 
    : dealii::FEValues<2>(fe, quadrature, update_flags),
      xquadrature_(nullptr),
      xdata_(nullptr)
  {
        xquadrature_ = dynamic_cast<XQuadratureBase*>(&quadrature);
        n_vertices_ = dealii::GeometryInfo<2>::vertices_per_cell;
  }
  
  using dealii::FEValues<2>::reinit;
  
  /** Reinits the values for the enriched cell (node values etc.)
   * Note that it calls the reinit function of the Deal II FEValues.
   */
  void reinit(XDataCell* xdata);

  const dealii::Point<2> &unit_quadrature_point (unsigned int i);   ///< Returns point on the unit cell.
  const dealii::Point<2> &real_quadrature_point (unsigned int i);   ///< Replaces the original method. 
  
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
//   dealii::Tensor<1,2> enrichment_grad(const unsigned int function_no, const unsigned int w, const dealii::Point<2> p);
  
private:
  /// Vector of values of the enrichment test function at quadrature points.
  std::vector<std::vector<double> > q_enrich_values_;
  
  /// Vector of the ramp function values (XFEM) at quadrature points.
  std::vector<std::vector<double> > q_ramp_values_;
  
  /// Function that precomputes XFEM ramp function or SGFEM interpolation at quadrature points.
  void prepare();
  
  XQuadratureBase * xquadrature_;
  XDataCell* xdata_;                                    ///< Pointer to enrichment data.
//   dealii::DoFHandler<2>::active_cell_iterator cell_;    ///< Cell iterator.
  unsigned int n_wells_;                                ///< Number of wells affecting the current cell.
  unsigned int n_vertices_;                             ///< Number of vertices per current cell.
};





/*******************************************     IMPLEMENTATION                   ***************************/

template<Enrichment_method::Type T>
inline const dealii::Point< 2 >& XFEValues<T>::real_quadrature_point(unsigned int i)
{
    if(xquadrature_) return xquadrature_->real_point(i);
    else return quadrature_point(i);
}

template<Enrichment_method::Type T>
inline const dealii::Point< 2 >& XFEValues<T>::unit_quadrature_point(unsigned int i)
{
    return this->get_quadrature().point(i);
}


template<Enrichment_method::Type T> 
void XFEValues<T>::reinit(XDataCell* xdata)
{
  xdata_ = xdata;
  n_wells_ = xdata_->n_wells();
  this->reinit(xdata_->get_cell());
  
  
  
  if((dealii::update_quadrature_points & update_flags) || xquadrature_)
  {
    if(dealii::update_values & update_flags)
    {
      q_enrich_values_.resize(n_wells_);
  
      for(unsigned int w=0; w < n_wells_; w++)
      {
        q_enrich_values_[w].resize(n_quadrature_points);

        for(unsigned int q=0; q < n_quadrature_points; q++)
          q_enrich_values_[w][q] = xdata_->get_well(w)->global_enrich_value(real_quadrature_point(q)); //returns quadrature_point in real coordinates
      }
      prepare();
    }
  }
}



#endif // XFEValues