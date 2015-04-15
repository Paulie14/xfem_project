#include "xquadrature_base.hh"
#include "system.hh"

#include "mapping.hh"

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/mapping.h>

const double XQuadratureBase::square_refinement_criteria_factor_ = 0.5;

const std::vector<QGauss<2> > XQuadratureBase::quadratures_ = {
        QGauss<2>(0),
        QGauss<2>(1),
        QGauss<2>(2),
        QGauss<2>(3),
        QGauss<2>(4),
        QGauss<2>(5),
        QGauss<2>(6),
        QGauss<2>(7),
        QGauss<2>(8),
        QGauss<2>(9),
        QGauss<2>(10),
        QGauss<2>(11),
        QGauss<2>(12),
        QGauss<2>(13),
        QGauss<2>(14)
    };
    
    
Square::Square(const Point< 2 > &p1, const Point< 2 > &p2)
  : refine_flag(false),
    processed(false),
    gauss(nullptr),
    transformed_to_real_(false)
{
  MASSERT( (p1[0] != p2[0]) && (p1[1] != p2[1]), "cannot create square");
  
  // numbering of the squares
  //      2
  //  3-------2
  //  |       |
  //3 |       | 1
  //  |       |
  //  0-------1
  //      0
  //
  
  //renumbering vertices - got 1 and 3
  if(p1[0] > p2[0] && p1[1] < p2[1])
  {
    vertices_[0] = Point<2>(p2[0], p1[1]);
    vertices_[1] = p1;
    vertices_[2] = Point<2>(p1[0], p2[1]);
    vertices_[3] = p2;
    mapping = MyMapping(vertices_[0],vertices_[2]);
    return;
  }
  
  //renumbering vertices - got 3 and 1
  if(p1[0] < p2[0] && p1[1] > p2[1])
  {
    vertices_[0] = Point<2>(p1[0], p2[1]);
    vertices_[1] = p2;
    vertices_[2] = Point<2>(p2[0], p1[1]);
    vertices_[3] = p1;
    mapping = MyMapping(vertices_[0],vertices_[2]);
    return;
  }
  
  //renumbering vertices - got 2 and 0
  if(p1[0] > p2[0])
  {
    vertices_[0] = p2;
    vertices_[1] = Point<2>(p1[0],p2[1]);
    vertices_[2] = p1;
    vertices_[3] = Point<2>(p2[0],p1[1]);
    mapping = MyMapping(vertices_[0],vertices_[2]);
    return;
  }
  
   //renumbering vertices - got 0 and 2
  vertices_[0] = p1;
  vertices_[1] = Point<2>(p2[0],p1[1]);
  vertices_[2] = p2;
  vertices_[3] = Point<2>(p1[0],p2[1]);
  
  unit_diameter_ = p1.distance(p2);
  mapping = MyMapping(vertices_[0],vertices_[2]);
}


void Square::transform_to_real_space(const DoFHandler< 2  >::active_cell_iterator& cell,
                                     const Mapping< 2 >& cell_mapping)
{
    if( ! transformed_to_real_) // if not already transformed
    {
        //mapping.print(cout);
        for(unsigned int i=0; i<4; i++)
        {
            real_vertices_[i] = cell_mapping.transform_unit_to_real_cell(cell,vertices_[i]); // map to real cell
        }
        real_diameter_ = std::max(real_vertices_[0].distance(real_vertices_[2]),
                                  real_vertices_[1].distance(real_vertices_[3]));
        //DBGMSG("unit_diameter_=%f, real_diameter_=%f\n", unit_diameter_, real_diameter_);
        transformed_to_real_ = true;
    }
}


XQuadratureBase::XQuadratureBase() 
: Quadrature< 2 >(),
    level_(0)
{}


void XQuadratureBase::apply_refinement(unsigned int n_squares_to_refine)
{
  //DBGMSG("refine() cell index = %d\n",cell->index());
  //temporary point - center of the square to refine
  Point<2> center;
  //counts nodes of a square that lie in the well
//   unsigned int n_nodes_in_well = 0;
  unsigned int n_original_squares = squares_.size();
  squares_.reserve(n_original_squares + 4*n_squares_to_refine);

  for(unsigned int i = 0; i < n_original_squares; i++)
  {
    if(squares_[i].refine_flag)
    {
      center = (squares_[i].vertex(0) + squares_[i].vertex(2)) / 2;
      for(unsigned int j = 0; j < 4; j++)
      {
        squares_.push_back(Square(squares_[i].vertex(j),center));
        squares_.back().gauss = squares_[i].gauss;
        
           //checking if the whole new square lies in the well
//         for(unsigned int w = 0; w < wells_.size(); w++)        
//         {
//             n_nodes_in_well = refine_criterion_nodes_in_well(squares_.back(),*(wells_[w]));
//             if (n_nodes_in_well == 4)
//             {
//                 //if the whole square lies in the well
//                 squares_.back().gauss = &(XQuadratureBase::quadratures_[0]);
//                 squares_.back().processed = true;
//             }
//             else
//             {
//                 //else it gets the quadrature from the descendant square
//                 squares_.back().gauss = squares_[i].gauss;
//             }
//         }
      }
    }
  }
  
  for(unsigned int i = 0; i < n_original_squares; i++)
  {
    if(squares_[i].refine_flag)
    {
      squares_.erase(squares_.begin()+i);
      i--;  //one erased, so we must lower iterator
    }
  }
  
  level_ ++;
  
  //print squares and nodes
  /*
  DBGMSG("Printing squares and nodes:\n");
  for(unsigned int i = 0; i < squares_.size(); i++)
  {
    std::cout << i << "\t";
    for(unsigned int j = 0; j < 4; j++)
      std::cout << squares_[i].vertices[j] << " | ";
    std::cout << std::endl;
  }
  //*/
}


void XQuadratureBase::gather_weights_points()
{
    if(quadrature_points.size() > 0) return; //do not do it again
    
    quadrature_points.reserve(squares_.size()*quadratures_[3].size());
    weights.reserve(squares_.size()*quadratures_[3].size());

//     if(squares_.size() == 1)
//     {
//         //DBGMSG("q_points: %d\n",squares[0].gauss->get_points().size());
//         quadrature_points = squares_[0].gauss->get_points();
//         weights = squares_[0].gauss->get_weights();
//     }
//     else
    {
        for(unsigned int i = 0; i < squares_.size(); i++)
        {   
            // if no quadrature points are to be added
            if(squares_[i].gauss == nullptr) continue;
            if( *(squares_[i].gauss) == XQuadratureBase::quadratures_[0]) continue;
            
            // map from unit square to unit cell
            std::vector<Point<2> > temp(squares_[i].gauss->get_points());
            squares_[i].mapping.map_unit_to_real(temp);  
            
            // gather vector of quadrature points and their weights
            for(unsigned int j = 0; j < temp.size(); j++)
            {
                quadrature_points.push_back(temp[j]);
                weights.push_back( squares_[i].gauss->weight(j) *
                                   squares_[i].mapping.jakobian() );
            }
        }
    }
    quadrature_points.shrink_to_fit();
    weights.shrink_to_fit();
    
    //DBGMSG("Number of quadrature points is %d\n",quadrature_points.size());
//     //control sum
//     #ifdef DEBUG    //----------------------
//     double sum = 0;
//     for(unsigned int i = 0; i < quadrature_points.size(); i++)
//     {
//         sum += weights[i];
//     }
//     sum = std::abs(sum-1.0);
//     if(sum > 1e-15) DBGMSG("Control sum of weights: %e\n",sum);
//     MASSERT(sum < 1e-12, "Sum of weights of quadrature points must be 1.0.\n");
//     #endif          //----------------------
}