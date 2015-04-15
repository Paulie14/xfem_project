#include <string>

#include <deal.II/base/point.h>

#include "adaptive_integration.hh"
#include "system.hh"
#include "data_cell.hh"
#include "xfevalues.hh"
#include "xquadrature_base.hh"

using namespace dealii;

Adaptive_integration::Adaptive_integration(XDataCell* xdata,
                                           const FE_Q< 2 >& fe, 
                                           XQuadratureBase* quadrature, 
                                           unsigned int m
                                          )
  : AdaptiveIntegrationBase(xdata, fe, quadrature, m)
{}




double Adaptive_integration::test_integration(Function< 2 >* func)
{
    DBGMSG("integrating...\n");
    double integral = 0;
//     gather_w_points();    //gathers all quadrature points from squares into one vector and maps them to unit cell
//     if (q_points_all.size() == 0) return 0;
    
//     Quadrature<2> quad(q_points_all, jxw_all);
    XFEValues<Enrichment_method::xfem_shift> xfevalues(*fe_,*xquad_, 
                                        update_quadrature_points | update_JxW_values );
    xfevalues.reinit(xdata_);
  
    for(unsigned int q=0; q<xquad_->size(); q++)
    {
//         if(q % 500 == 0) DBGMSG("q: %d\n",q);
        integral += func->value(xfevalues.quadrature_point(q)) * xfevalues.JxW(q);
    }
//     DBGMSG("q done\n");
    return integral;
}


// std::pair< double, double > Adaptive_integration::test_integration_2(Function< 2 >* func, unsigned int diff_levels)
// {   
//     QGauss<2> gauss_quad = Adaptive_integration::quadratures_[1];
//     
//     // Prepare quadrature points - only on squares on the well edge
//     MASSERT(squares.size() > 1, "Element not refined.");
//     q_points_all.reserve(squares.size()*gauss_quad.size());
//     jxw_all.reserve(squares.size()*gauss_quad.size());
// 
//     for(unsigned int i = 0; i < squares.size(); i++)
//     {   
//         //TODO: this will not include the squares with cross-section with well and no nodes inside
//         //all the squares on the edge of the well
//         bool on_well_edge = false;
//         for(unsigned int w=0; w < xdata->n_wells(); w++)
//         {
//             unsigned int n = refine_criterion_nodes_in_well(squares[i],*(xdata->get_well(w)));
//             if( (n > 0) && (n < 4)) on_well_edge = true;
//         }
//         if(on_well_edge)    
//         {
//             squares[i].gauss = &(gauss_quad);
//             std::vector<Point<2> > temp(squares[i].gauss->get_points());
//             squares[i].mapping.map_unit_to_real(temp);  //mapped from unit square to unit cell
// 
//             for(unsigned int w=0; w < xdata->n_wells(); w++)
//             for(unsigned int j = 0; j < temp.size(); j++)
//             {
//                 //include only points outside the well
//                 Point<2> real_quad = mapping->transform_unit_to_real_cell(cell, temp[j]);
//                 if(xdata->get_well(w)->center().distance(real_quad) >= xdata->get_well(w)->radius())
//                 {
//                 q_points_all.push_back(temp[j]);
//                 jxw_all.push_back( squares[i].gauss->weight(j) *
//                                    squares[i].mapping.jakobian() );
//                 }
//             }
//             squares[i].refine_flag = false;
//         }
//         //all the squares around the well
//         else
//         {
//             squares[i].refine_flag = true;
//             //nothing - we ignore other squares
//         }
//     }
//     q_points_all.shrink_to_fit();
//     jxw_all.shrink_to_fit();
//     
// //     gnuplot_refinement("../output/test_adaptive_integration_2/",true);
// //     squares.erase(std::remove_if(squares.begin(), squares.end(),remove_square_cond), squares.end());
//     for (auto it = squares.begin(); it != squares.end(); ) {
//         if (it->refine_flag)
//             // new erase() that returns iter..
//             it = squares.erase(it);
//         else
//             ++it;
//     }
//  
// //     gnuplot_refinement("../output/test_adaptive_integration_2/",true);
//     
//     DBGMSG("squares refined and prepared...\n");
//     
//     Quadrature<2>* quad = new Quadrature<2>(q_points_all, jxw_all);
//     XFEValues<Enrichment_method::xfem_shift> *xfevalues = 
//         new XFEValues<Enrichment_method::xfem_shift>(*fe,*quad, update_quadrature_points | update_JxW_values );
//     xfevalues->reinit(xdata);
//   
//     double integral_test = 0;
//     for(unsigned int q=0; q<q_points_all.size(); q++)
//     {
//         integral_test += func->value(xfevalues->quadrature_point(q)) * xfevalues->JxW(q);
//     }
// 
//     DBGMSG("test integral computed...%f\n",integral_test);
//     
//     //refine several times
//     for(unsigned int i = 0; i < diff_levels; i++)
//     {
//         unsigned int squares_to_refine = squares.size();
//         DBGMSG("refining...\n");
//         for(unsigned int i = 0; i < squares.size(); i++)
//             squares[i].refine_flag = true;
//         refine(squares_to_refine);
//     }
//         
//     //gnuplot_refinement("../output/test_adaptive_integration_2/",true);
//     // Prepare quadrature points - only on squares on the well edge
//     MASSERT(squares.size() > 1, "Element not refined.");
//     q_points_all.clear();
//     jxw_all.clear();
//     q_points_all.reserve(squares.size()*quadratures_[1].size());
//     jxw_all.reserve(squares.size()*quadratures_[1].size());
// 
//     for(unsigned int i = 0; i < squares.size(); i++)
//     {   
//             squares[i].gauss = &(gauss_quad);
//             std::vector<Point<2> > temp(squares[i].gauss->get_points());
//             squares[i].mapping.map_unit_to_real(temp);  //mapped from unit square to unit cell
// 
//             for(unsigned int w=0; w < xdata->n_wells(); w++)
//             for(unsigned int j = 0; j < temp.size(); j++)
//             {
//                 //include only points outside the well
//                 Point<2> real_quad = mapping->transform_unit_to_real_cell(cell, temp[j]);
//                 if(xdata->get_well(w)->center().distance(real_quad) >= xdata->get_well(w)->radius())
//                 {
//                 q_points_all.push_back(temp[j]);
// //                 DBGMSG("i=%d \t w=%f \t j=%f\n",i,squares[i].gauss->weight(j), squares[i].mapping.jakobian());
//                 jxw_all.push_back( squares[i].gauss->weight(j) *
//                                    squares[i].mapping.jakobian() );
//                 }
//             }
//     }
//     q_points_all.shrink_to_fit();
//     jxw_all.shrink_to_fit();
//     
//     //gnuplot_refinement("../output/test_adaptive_integration_2/",true);
//     DBGMSG("squares refined and prepared...\n");
//     
//     delete xfevalues;
//     delete quad;
//     
//     quad = new Quadrature<2>(q_points_all, jxw_all);
//     xfevalues = 
//         new XFEValues<Enrichment_method::xfem_shift>(*fe,*quad, update_quadrature_points | update_JxW_values );
//     xfevalues->reinit(xdata);
//     double integral_fine = 0;
//     for(unsigned int q=0; q<q_points_all.size(); q++)
//     {
// //         std::cout << xfevalues->quadrature_point(q) << std::endl;
// //         DBGMSG("q=%d \t jxw=%f\n",q,xfevalues->JxW(q));
//         integral_fine += func->value(xfevalues->quadrature_point(q)) * xfevalues->JxW(q);
//     }
// //     cout << jxw_all[0] << "  " << q_points_all[10] <<endl;
//     DBGMSG("fine integral computed...%f\n",integral_fine);
//     
//     delete xfevalues;
//     delete quad;
//     
//     return std::make_pair(integral_test, integral_fine);
// }





SmoothStep::SmoothStep(Well* well, double band_width)
:  Function< 2 >(),
    well_(well),
    band_width_(band_width),
    coefs_{6, -15, 10, 0, 0, 0},  // fifth degree smooth step polynomial
    size_(6)
{}

double SmoothStep::value(const Point< 2 >& p, const unsigned int component) const
{
    double r = well_->center().distance(p);
    return value(r);
}

double SmoothStep::value(const double& r) const
{
    if (r < well_->radius()) return 0.0;      // inside the well
    if (r > well_->radius() + band_width_) return 1.0;          // outside the band
        
    // map r onto [0,1] interval
    double x = (r - well_->radius()) / band_width_;
//     DBGMSG("x=%f\n",x);
    double res = coefs_[0];
    for(unsigned int i=1; i<size_; i++)
        res = res*x + coefs_[i];       //polynomial evaluation by Horner
    
    return res;
}


AdaptiveIntegrationPolar::AdaptiveIntegrationPolar(XDataCell * xdata, 
                                                   const FE_Q< 2, 2 >& fe, 
                                                   XQuadratureBase* quadrature, 
                                                   std::vector< XQuadratureWell* > polar_xquads,
                                                   unsigned int m)
: AdaptiveIntegrationBase(xdata, fe, quadrature, m),
    polar_xquads_(polar_xquads)
{}

