#ifndef XQUADRATURE_WELL_H
#define XQUADRATURE_WELL_H

#include <deal.II/base/point.h>

#include "xquadrature_base.hh"

//forward declaration
namespace dealii{
        template<int,int> class Mapping;
}
template<int dim,int spacedim=dim> using DealMapping = dealii::Mapping<dim,spacedim>;

class Well;
class XDataCell;

namespace dealii{
    template<int,int> class Mapping;
}

class XQuadratureWell : public XQuadratureBase
{
public:
    XQuadratureWell();
    
    XQuadratureWell(Well * well, double width);
    
    /// Getter for vector of quadrature points in real coordinates.
    const std::vector<Point<2> > & polar_points();
    
    /// Getter for i-th quadrature point in real coordinates.
    const Point<2> & polar_point(unsigned int i);
    
    /// Creates refinement of a cell -- new quadrature.
    void refine(unsigned int max_level) override;
    
    /** @brief Calls gnuplot to create image of refined element.
     * 
      * Also can save the gnuplot script to file.
      * @param output_dir is the directory for output_dir
      * @param real is true then the element is printed in real coordinates
      * @param show is true then the gnuplot utility is started and plots the refinement on the screen
      */ 
    void gnuplot_refinement(const std::string &output_dir, bool real=true, bool show=false) override;
    
    // Create subquadrature with only the quadrature points that lie in the given cell.
    void create_subquadrature(XQuadratureWell & new_xquad, 
                              const dealii::DoFHandler< 2  >::active_cell_iterator& cell,
                              const DealMapping<2> & mapping
                             );
    
private:
    
    /// Returns true if criterion is satisfied.
    /** Criterion: square diameter > C * (minimal distance of a node from well edge)
     */
    bool refine_criterion_a(Square &square);
        
    bool refine_error();
    
    /// Gathers the quadrature points and their weigths from squares into a single vector.
    void gather_weights_points() override;
    
    void map_polar_quadrature_points_to_real(void);
    
    Point<2> map_from_polar(Point<2> point);
    Point<2> map_into_polar(Point<2> point);
    
    void transform_square_to_real(Square & square);
    
    /// Quadrature around well.
    Well * well_;
    
    /// Width of the band around the well.
    double width_;
    
    std::vector<Point<2> > polar_quadrature_points_;
    
    friend Square;
};



/********************************           IMPLEMENTATION                  *********************************/
#include "system.hh"

inline const Point< 2 >& XQuadratureWell::polar_point(unsigned int i)
{   MASSERT(i < real_points_.size(), "Index 'i' exceeded size of vector of quadrature points.");
    return polar_quadrature_points_[i];
}

inline const vector< Point< 2 > >& XQuadratureWell::polar_points()
{   return polar_quadrature_points_;
}




#endif  //XQUADRATURE_WELL_H