#ifndef Mapping_h
#define Mapping_h

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

using namespace dealii;

/** @brief Class that maps points between squares (see @p Square). 
 * 
 * Simple mapping between 2d rectangular objects (in fact used only for squares)
 * This class is used in @p Adaptive_integration to map refined squares to the reference cell.
 */
class MyMapping
{
public:
  ///@brief Default constructor.
  MyMapping()
  {}
  
  /** @brief Constructor
    * @param p1 is down left vertex of the square
    * @param p2 is up right vertex of the square
    */ 
  MyMapping(const Point<2> &p1, const Point<2> &p2);
  
  ///Maps point from real and unit cell.
  Point<2> map_real_to_unit(Point<2> point);
  ///Maps point from real and unit cell.
  Point<2> map_unit_to_real(Point<2> point);
  
  ///Scale vector from unit cell to square.
  Tensor<1,2> scale_inverse(Tensor<1,2> vec);
  ///Scale vector from square to unit cell.
  Tensor<1,2> scale(Tensor<1,2> vec);
    
  ///maps between unit and real cell
  void map_unit_to_real(std::vector<Point<2> > &points);
  
  ///maps between real and unit cell
  void map_real_to_unit(std::vector<Point<2> > &points);
  
  ///maps between unit and real cell
  void map_unit_to_real(const std::vector<Point<2> > &points, std::vector<Point<2> > &points_out);
  
  ///maps between real and unit cell
  void map_real_to_unit(const std::vector<Point<2> > &points, std::vector<Point<2> > &points_out);
  
  ///returns jacobian of mapping from real to unit
  inline double jakobian()
  { return scale_matrix[0][0] * scale_matrix[1][1];}
  
  ///returns jacobian of mapping from unit to real
  inline double jakobian_inv()
  { return 1/jakobian();}
  
  ///prints itself
  void print(std::ostream &output);
  
private:
  ///scale from unit to real
  Tensor<2,2,double> scale_matrix;
  ///scale from real to unit
  Tensor<2,2,double> scale_inv_matrix;
  ///translation from unit to real;
  Tensor<1,2,double> translation_vec;
};

#endif  //Mapping_h