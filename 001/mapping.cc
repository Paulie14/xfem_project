#include "mapping.hh"
#include "system.hh"

MyMapping::MyMapping(const dealii::Point< 2 >& p1, const dealii::Point< 2 >& p2)
{
  scale_matrix = Tensor<2,2>();
  scale_inv_matrix = Tensor<2,2>();
  scale_inv_matrix[0][0] = 1/ (p2[0] - p1[0]);
  scale_inv_matrix[1][1] = 1/ (p2[1] - p1[1]);
  scale_matrix[0][0] = p2[0] - p1[0];
  scale_matrix[1][1] = p2[1] - p1[1];
    
  translation_vec = p1;
}

Point< 2 > MyMapping::map_real_to_unit(Point< 2 > point)
{
  point = point - translation_vec;
  point = scale_inv_matrix * point;
  return point;
}

Point< 2 > MyMapping::map_unit_to_real(Point< 2 > point)
{
  point = scale_matrix * point;
  point = point + translation_vec;
  return point;
}

Tensor< 1, 2 > MyMapping::scale_inverse(Tensor< 1, 2 > vec)
{
  vec = scale_inv_matrix * vec;
  return vec;
}

Tensor< 1, 2 > MyMapping::scale(Tensor< 1, 2 > vec)
{
  vec = scale_matrix * vec;
  return vec;
}

void MyMapping::map_unit_to_real(std::vector< Point< 2 > >& points)
{
  for(unsigned int i=0; i < points.size(); i++)
  {
    points[i] = scale_matrix * points[i];
    points[i] = translation_vec + points[i];
  }
}

void MyMapping::map_real_to_unit(std::vector< Point< 2 > >& points)
{
  for(unsigned int i=0; i < points.size(); i++)
  {
    points[i] = points[i] - translation_vec;
    points[i] = scale_inv_matrix * points[i];
  }
}

void MyMapping::map_unit_to_real(const std::vector< Point< 2 > >& points, std::vector<Point<2> > &points_out)
{
  points_out.resize(points.size());
  for(unsigned int i=0; i < points.size(); i++)
  {
    points_out[i] = scale_matrix * points[i];
    points_out[i] = translation_vec + points[i];
  }
}

void MyMapping::map_real_to_unit( const std::vector< Point< 2 > >& points, std::vector<Point<2> > &points_out)
{
  points_out.resize(points.size());
  for(unsigned int i=0; i < points.size(); i++)
  {
    points_out[i] = points[i] - translation_vec;
    points_out[i] = scale_inv_matrix * points[i];
  }
}

void MyMapping::print(std::ostream &output)
{
  output << "mapping: scale: " << scale_matrix << "\t translation: " << translation_vec
         << "\t jakobian: " << jakobian() << "\n";
}