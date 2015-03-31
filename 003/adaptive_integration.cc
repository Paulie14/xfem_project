
#include <string>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <boost/graph/graph_concepts.hpp>

#include "adaptive_integration.hh"
#include "system.hh"
#include "gnuplot_i.hpp"
#include "data_cell.hh"
#include "mapping.hh"
#include "xfevalues.hh"

const vector<double> Adaptive_integration::alpha_ = 
    {   1.00000000000000e+00,   
        1.66666666666667e-01,   
        2.00000000000000e-02,
        2.04081632653061e-03,
        1.88964474678760e-04,
        1.63977436704709e-05,
        1.35839296678458e-06,
        1.08671437342766e-07,
        8.46057903187626e-09,
        6.44503942871459e-10,
        4.82281862012653e-11,
        3.55557516417654e-12,
        2.58845871952048e-13,
        1.86411636179457e-14,
        1.32992843885462e-15
    };
const double Adaptive_integration::c_empiric_ = 12.65;
const double Adaptive_integration::p_empiric_ = 2.27;
const double Adaptive_integration::square_refinement_criteria_factor_ = 0.5;
// const double Adaptive_integration::square_refinement_criteria_factor_ = 1.0;

const std::vector<QGauss<2> > Adaptive_integration::quadratures_ = {
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



Adaptive_integration::Adaptive_integration(const DoFHandler< 2  >::active_cell_iterator& cell, 
                                           const dealii::FE_Q< 2 >& fe,
                                           const Mapping<2>& mapping,
                                           unsigned int m
                                          )
  : cell(cell), fe(&fe), mapping (&mapping), m_(m),
    dirichlet_function(nullptr),
    rhs_function(nullptr),
    level_(0)
{
      MASSERT(cell->user_pointer() != NULL, "NULL user_pointer in the cell"); 
      //A *a=static_cast<A*>(cell->user_pointer()); //from DEALII (TriaAccessor)
      xdata = static_cast<XDataCell*>( cell->user_pointer() );
      //xdata->initialize();
      
      //first square
      squares.push_back(Square(Point<2>(0,0), Point<2>(1,1)));
      
      for(unsigned int w = 0; w < xdata->n_wells(); w++)
      {
//         //testing if the well is inside
//         Triangulation<2>::active_face_iterator face;
//         for(unsigned int face_no = 0; face_no < xdata->n_wells(); face_no++)
//         {
//             face = cell->face(face_no);
//             Point<2> a = face->vertex(0),
//                      b = face->vertex(1);
//             Tensor<1,2> direction_vector; 
//             direction_vector[0] = b[0] - a[0];  //x coordinate
//             direction_vector[1] = b[1] - a[1]; //y coordinate
//             
//             double t1,t2;   //line parameter
//             double aa = direction_vector[0] * direction_vector[0]
//                         + direction_vector[1] * direction_vector[1],
//                    bb = 2 * (direction_vector[0]*a[0] + direction_vector[1]*a[1]),
//                    cc = a[0]*a[0] + a[1]*a[1] - xdata->get_well(w)->radius()*xdata->get_well(w)->radius();
//                    
//             double discriminant = bb*bb - 4*aa*cc;
//             
//             if(discriminant >= 0)
//             {
//                 t1 = (-bb - std::sqrt(discriminant)) / (2*aa);
//                 t2 = (-bb + std::sqrt(discriminant)) / (2*aa);
//                 DBGMSG("well_inside: t1=%f \t t2=%f\n",t1, t2);
//                 if( (t1 >= 0) && (t1 <= 1) ) well_inside[w] = true;
//                 if( (t2 >= 0) && (t2 <= 1) ) well_inside[w] = true;
//             }
//             
//         }
          
        //if the whole square is inside the well
        if(refine_criterion_nodes_in_well(squares[0],*(xdata->get_well(w))) == 4)
        {
            squares[0].gauss = &(Adaptive_integration::quadratures_[0]);
            squares[0].processed = true;
        }
      }
}

bool Adaptive_integration::refine_criterion_a(Square& square, Well& well)
{
    //return false; // switch on and off the criterion
    square.transform_to_real_space(cell, *mapping);
    
    double min_distance = square.real_vertex(0).distance(well.center());// - well.radius();
    for(unsigned int j=1; j < 4; j++)
    {
        double dist = well.center().distance(square.real_vertex(j));// - well.radius();
        min_distance = std::min(min_distance,dist);
    }

    //DBGMSG("square [%d] diameter=%f , min_distance=%f cell_diameter=%f\n",i,squares[i].real_diameter(),min_distance, cell->diameter());
    // criteria:
    if( square.real_diameter() > square_refinement_criteria_factor_ * min_distance)
        return true;
    else return false;
}

unsigned int Adaptive_integration::refine_criterion_nodes_in_well(Square& square, Well& well)
{
    square.transform_to_real_space(cell, *mapping);

    unsigned int vertices_in_well = 0;
    for(unsigned int j=0; j < 4; j++)
    {
        if(well.points_inside(square.real_vertex(j)))
            vertices_in_well++;
    }
    //DBGMSG("nodes in well %d\n",vertices_in_well);
    return vertices_in_well;
}


bool Adaptive_integration::refine_edge()
{
    Tensor<1,2> direction_vectors[4];
    double lines_parameters[4];
    double distances[4];
    Point<2> t_points[4];
    unsigned int n_squares_to_refine = 0; 
    unsigned int n_nodes_in_well;
    Well* well;
  /* there are several cases that can happen:
   * 1] a node of a square can be inside a well       -> refine
   * 2] if all nodes of a square are inside a well 
   *    the whole square is inside                    -> no refine
   * 3] if no node of a square is inside a well 
   *    and the center of a well is inside the square 
   *    then the whole well is inside the square      -> refine
   * 4] if no nodes are inside are in a well
   *    and the well edge crosses the square line     -> refine
   * 5] if the minimum distance of square vertex 
   *    to well edge is smaller than square diameter  -> refine
   */
  //DBGMSG("wells.size(): %d", xdata->wells().size());
  
    for(unsigned int i = 0; i < squares.size(); i++)
    { 
        if(squares[i].processed) 
        {
            //DBGMSG("processed\n");
            continue;
        }
                
        for(unsigned int w = 0; w < xdata->n_wells(); w++)
        {
            well = xdata->get_well(w);
            //skip squares that are already flagged
            if(squares[i].refine_flag) continue;
        
            // refinement on the cells without well inside
            if( xdata->q_points(w).size() == 0)    // is the well not inside ? )
            {
                squares[i].gauss = &(Adaptive_integration::quadratures_[3]);
                //minimum distance from well criterion      ------------------------------------[5]
                if( refine_criterion_a(squares[i],*well) )
                {
                    squares[i].refine_flag = true;
                    n_squares_to_refine++;
                }
            }
            else
            {
                //testing the edge of the well
                //by the distance of a point (center of the well) to a line (side of the square)
      
                n_nodes_in_well = refine_criterion_nodes_in_well(squares[i],*well);

                //if the whole square is not inside the the well      ------------------------------------[1]
                if(n_nodes_in_well !=0 && n_nodes_in_well < 4)
                {
                    //squares on the edge of the well obtain three point quadrature
                    squares[i].gauss = &(Adaptive_integration::quadratures_[3]);
                    //std::cout << i << " addded(node)\n";
                    n_squares_to_refine++;
                    squares[i].refine_flag = true;
                    continue;
                }
                //if the whole square is inside the well              ------------------------------------[2]
                if (n_nodes_in_well == 4) 
                {
                    //squares inside the well obtain one point quadrature
                    squares[i].gauss = &(Adaptive_integration::quadratures_[0]);
                    continue;
                }
      
                // temporary shortcuts
                const Point<2>* vertices = squares[i].real_vertices();
                Point<2> well_center = well->center();
                
                //if the whole well is inside the square              ------------------------------------[3]
                if ( n_nodes_in_well == 0 
                    && well_center[0] >= vertices[0][0]
                    && well_center[0] <= vertices[2][0]
                    && well_center[1] >= vertices[0][1]
                    && well_center[1] <= vertices[2][1]
                    ) 
                {
                    //squares outside the well obtain three point quadrature
                    squares[i].gauss = &(Adaptive_integration::quadratures_[3]);
                    n_squares_to_refine++;
                    squares[i].refine_flag = true;
                    continue;
                }
                
                //computing lines:
                //if there are no nodes of the square in the circle of the well
                //then if the line goes through the well edge 
                //check if the sum of distances of neighbour lines from center is equal the side of square
                for(unsigned int j = 0; j < 3; j++)
                {
                    direction_vectors[j] = vertices[j+1] - vertices[j];
                    lines_parameters[j] = ( direction_vectors[j] * 
                                            (well_center - vertices[j])
                                        ) / direction_vectors[j].norm_square();
                                        
                    t_points[j] = lines_parameters[j]*direction_vectors[j] + vertices[j];
                    distances[j] = well_center.distance(t_points[j]);
                }
                direction_vectors[3] = vertices[0] - vertices[3];
                lines_parameters[3] = (direction_vectors[3] * (well_center - vertices[3])) 
                                      / direction_vectors[3].norm_square();
                t_points[3] = lines_parameters[3]*direction_vectors[3] + vertices[3];
                distances[3] = well_center.distance(t_points[3]);
                
                //std::cout << "distance\t";
                for(unsigned int j = 0; j < 4; j++)
                {
                    //std::cout << distances[j] << " _ ";
                    if(distances[j] <= well->radius())
                    {
                    int a = j-1,
                        b = j+1;
                    if (j == 0) 
                        a = 3;
                    if (j == 3) 
                        b = 0;
                    
                    //then the well edge crosses the square line------------------------------------[4]
                    if( std::abs(distances[a] + distances[b] - 
                                    vertices[0].distance(vertices[1])) < 1e-13) 
                    {
                        //squares on the edge of the well obtain three point quadrature
                        squares[i].gauss = &(Adaptive_integration::quadratures_[3]);
                        //std::cout << i << " addded(line)\n";
                        n_squares_to_refine++;
                        squares[i].refine_flag = true;
                        break;
                    }
                    }
                }

                if(squares[i].refine_flag) continue;
                
                //minimum distance from well criterion      ------------------------------------[5]
                if( refine_criterion_a(squares[i],*well) )
                {
                        squares[i].gauss = &(Adaptive_integration::quadratures_[3]);
                        squares[i].refine_flag = true;
                        n_squares_to_refine++;
                }
                //std::cout << std::endl;
            }   // if
        } // for w
        
        if(! squares[i].refine_flag) squares[i].processed = true;
    } // for i
  
  if (n_squares_to_refine == 0) 
    return false;
  else
  {
    refine(n_squares_to_refine);
    return true;
  }
}

unsigned int Adaptive_integration::refine_criterion_alpha(double r_min)
{
    double r_min_square = r_min*r_min;
    unsigned int fail_result = -1;
    double rhs = alpha_tolerance_/2.0*r_min_square;
    double pow_half = 1.0;  
    for(unsigned int i=0; i<alpha_.size(); i++)
    {
        pow_half = pow_half * square_refinement_criteria_factor_*square_refinement_criteria_factor_; // 0.5^(2i)
        double val = alpha_[i]*pow_half;
        //DBGMSG("alpha criterion: val=%e, rhs=%e\n", val, rhs);
        if( val <= rhs) return i+1;
    }
    return fail_result;
}

bool Adaptive_integration::refine_criterion_h(Square& square, Well& well, double criterion_rhs)
{   
    //DBGMSG("h criterion: h=%e, rhs=%e\n",square.real_diameter(), criterion_rhs);
    return (square.real_diameter() > criterion_rhs);
}

double Adaptive_integration::compute_r_min(Square& square, unsigned int w)
{
    Well well = *(xdata->get_well(w));
    Point<2> wc = well.center();
    std::vector<unsigned int> quadrants(4);
    for(unsigned int j = 0; j < 4; j++)
    {
        int x = (square.real_vertex(j)(0)-wc(0)) >= 0;
        int y = (square.real_vertex(j)(1)-wc(1)) >= 0;
        quadrants[j] =  3 + x - y - 2 * x * y;
    }
    //DBGMSG("quadrants: %d%d%d%d\n",quadrants[0], quadrants[1], quadrants[2], quadrants[3]);
    Tensor<1,2> direction_vector;
    unsigned int a,b;
    double lines_parameter;
    Point<2> t_point;
    if( (quadrants[0] != quadrants[1]))
    {
        a = quadrants[1]-1;
        b = (a == 0) ? 1 : 2;
    }
    else if( (quadrants[1] != quadrants[2]))
    {
        a = quadrants[2]-1;
        b = (a == 0) ? 3 : 2;
    }
    else    // all points are in the same quadrants
    {
        //quadrant corresponds with the node numbering
        return square.real_vertex(quadrants[0]-1).distance(well.center());// - well.radius();
    }
        
    direction_vector = square.real_vertex(b) - square.real_vertex(a);
    lines_parameter = ( direction_vector * 
                                (wc - square.real_vertex(a))
                            ) / direction_vector.norm_square();
                            
    t_point = lines_parameter*direction_vector + square.real_vertex(a);
    //Point<2> t_point_real = mapping->transform_unit_to_real_cell(cell,t_point);
    //return well.center().distance(t_point_real);
    return well.center().distance(t_point);// - well.radius();
}

bool Adaptive_integration::refine_error(double alpha_tolerance)
{
    unsigned int n_squares_to_refine = 0; 
    unsigned int n_nodes_in_well;
    Well well;
    
    alpha_tolerance_ = alpha_tolerance;
    double h_criterion_rhs;
    
    /* there are several cases that can happen:
    * 1] a node of a square can be inside a well       -> refine
    * 2] if all nodes of a square are inside a well 
    *    the whole square is inside                    -> no refine
    * 3] if no node of a square is inside a well 
    *    and the center of a well is inside the square 
    *    then the whole well is inside the square      -> refine
    * 4] if no nodes are inside are in a well
    *    and the well edge crosses the square line     -> refine
    * 5] if the minimum distance of square vertex 
    *    to well edge is smaller than square diameter  -> refine
    */
    //DBGMSG("wells.size(): %d", xdata->wells().size());
  
    for(unsigned int i = 0; i < squares.size(); i++)
    { 
        //DBGMSG("square[%d]:\n",i);
        if(squares[i].processed) 
        {
            //DBGMSG("processed\n");
            continue;
        }
                
        for(unsigned int w = 0; w < xdata->n_wells(); w++)
        {
            well = *(xdata->get_well(w));
            h_criterion_rhs = std::pow(alpha_tolerance_ / c_empiric_ * std::pow(well.radius(),p_empiric_), 
                                       1.0/(p_empiric_-1.0)); 
            //skip squares that are already flagged
            if(squares[i].refine_flag) continue;
        
            // refinement on the cells without well inside
//             if( xdata->q_points(w).size() == 0)    // is the well not inside ? )
//             {
//                 //minimum distance from well criterion      ------------------------------------[5]
//                 if( refine_criterion_a(squares[i],*well) )
//                 {
//                     squares[i].refine_flag = true;
//                     n_squares_to_refine++;
//                 }
//             }
//             else
            {
                //testing the edge of the well
                //by the distance of a point (center of the well) to a line (side of the square)
      
                n_nodes_in_well = refine_criterion_nodes_in_well(squares[i], well);

                //if the whole square is inside the well              ------------------------------------[1]
                if (n_nodes_in_well == 4) 
                {
                    //squares inside the well obtain no quadrature points
                    //DBGMSG("square[%d]: inside well\n", i);
                    squares[i].gauss = &(Adaptive_integration::quadratures_[0]);
                    continue;
                }
                
                //if the whole square is not inside the the well and criterion h -------------------------[2]
                if(n_nodes_in_well !=0 && n_nodes_in_well < 4)
                {
                    //DBGMSG("square[%d]: node in well\n",i);
                    if(refine_criterion_h(squares[i], well, h_criterion_rhs))
                    {
                        n_squares_to_refine++;
                        squares[i].refine_flag = true;
                    }
                    else
                    {
                        squares[i].gauss = &(Adaptive_integration::quadratures_[1]);
                        //squares[i].refine_flag = false;
                    }
                    continue;
                }

      
                //if the whole well is inside the square              ------------------------------------[3]
                if ( n_nodes_in_well == 0 
                    && (well.center()[0] >= squares[i].real_vertex(0)[0]) 
                    && (well.center()[0] <= squares[i].real_vertex(2)[0])
                    && (well.center()[1] >= squares[i].real_vertex(0)[1]) 
                    && (well.center()[1] <= squares[i].real_vertex(2)[1]) 
                    ) 
                {
                    //DBGMSG("square[%d]: well inside the square\n", i);
                    //squares outside the well obtain three point quadrature
                    //squares[i].gauss = &(Adaptive_integration::gauss_3);
                    n_squares_to_refine++;
                    squares[i].refine_flag = true;
                    continue;
                }
                
                double r_min = compute_r_min(squares[i], w);
                //DBGMSG("r_min = %e\n",r_min);
                double val = squares[i].real_diameter() / r_min;
                if(val > square_refinement_criteria_factor_)
                {
                    //DBGMSG("square[%d]: refine: h/rmin = %d\n",i, val);
                    n_squares_to_refine++;
                    squares[i].refine_flag = true;
                    continue;
                }
                else
                {
                    unsigned int quad_order = refine_criterion_alpha(r_min);
                    if(quad_order < Adaptive_integration::quadratures_.size())
                    {
                        //DBGMSG("square[%d]:\n",i);
                        if (quad_order == 1) quad_order = 2;
                        squares[i].gauss = &(Adaptive_integration::quadratures_[quad_order]);
                    }
                    else
                    {
                        //DBGMSG("square[%d]: refine: alpha criterion failed\n", i);
                        n_squares_to_refine++;
                        squares[i].refine_flag = true;
                    }
                    continue;
                }
                
            }   // if
        } // for w
        
        if(! squares[i].refine_flag) squares[i].processed = true;
    } // for i
  
  if (n_squares_to_refine == 0) 
    return false;
  else
  {
    refine(n_squares_to_refine);
    return true;
  }
}

void Adaptive_integration::refine(unsigned int n_squares_to_refine)
{
  //DBGMSG("refine() cell index = %d\n",cell->index());
  //temporary point - center of the square to refine
  Point<2> center;
  //counts nodes of a square that lie in the well
  unsigned int n_nodes_in_well = 0;
  unsigned int n_original_squares = squares.size();
  squares.reserve(n_original_squares + 4*n_squares_to_refine);
  
  for(unsigned int i = 0; i < n_original_squares; i++)
  {
    if(squares[i].refine_flag)
    {
      center = (squares[i].vertex(0) + squares[i].vertex(2)) / 2;
      for(unsigned int j = 0; j < 4; j++)
      {
        squares.push_back(Square(squares[i].vertex(j),center));
        
        for(unsigned int w = 0; w < xdata->n_wells(); w++)        
        {
            //checking if the whole new square lies in the well
            n_nodes_in_well = refine_criterion_nodes_in_well(squares.back(),*(xdata->get_well(w)));
            if (n_nodes_in_well == 4)
            {
                //if the whole square lies in the well
                squares.back().gauss = &(Adaptive_integration::quadratures_[0]);
                squares.back().processed = true;
            }
            else
            {
                //else it gets the quadrature from the descendant square
                squares.back().gauss = squares[i].gauss;
            }
        }
      }
    }
  }
  
//   squares.erase(std::remove_if(squares.begin(), squares.end(),remove_square_cond));
  for(unsigned int i = 0; i < n_original_squares; i++)
  {
    if(squares[i].refine_flag)
    {
      squares.erase(squares.begin()+i);
      i--;  //one erased, so we must lower iterator
    }
  }
  
  level_ ++;
  
  //print squares and nodes
  /*
  DBGMSG("Printing squares and nodes:\n");
  for(unsigned int i = 0; i < squares.size(); i++)
  {
    std::cout << i << "\t";
    for(unsigned int j = 0; j < 4; j++)
      std::cout << squares[i].vertices[j] << " | ";
    std::cout << std::endl;
  }
  //*/
}


void Adaptive_integration::gather_w_points()
{
    if(q_points_all.size() > 0) return; //do not do it again
    
    q_points_all.reserve(squares.size()*quadratures_[3].size());
    jxw_all.reserve(squares.size()*quadratures_[3].size());

    if(squares.size() == 1)
    {
        //DBGMSG("q_points: %d\n",squares[0].gauss->get_points().size());
        q_points_all = squares[0].gauss->get_points();
        jxw_all = squares[0].gauss->get_weights();
    }
    else
    {
        for(unsigned int i = 0; i < squares.size(); i++)
        {   
            // if no quadrature points are to be added
            if(squares[i].gauss == nullptr) continue;
            if( *(squares[i].gauss) == Adaptive_integration::quadratures_[0]) continue;
            
            // TODO: try to save the mapped quad point for later usage 
            // (perhaps, put in xshape value both unit and real point)
            std::vector<Point<2> > temp(squares[i].gauss->get_points());
            squares[i].mapping.map_unit_to_real(temp);  //mapped from unit square to unit cell
            
            for(unsigned int j = 0; j < temp.size(); j++)
            {
                bool include_point = true;
                for(unsigned int w=0; w < xdata->n_wells(); w++)
                {
                    //include only points outside the well
                    Point<2> real_quad = mapping->transform_unit_to_real_cell(cell, temp[j]);
                    if(xdata->get_well(w)->center().distance(real_quad) <= xdata->get_well(w)->radius())
                        include_point = false;
                }
                if(include_point)
                {
                    q_points_all.push_back(temp[j]);
                    jxw_all.push_back( squares[i].gauss->weight(j) *
                                    squares[i].mapping.jakobian() );
                }
            }
        }
    }
    q_points_all.shrink_to_fit();
    jxw_all.shrink_to_fit();
    
    //DBGMSG("Number of quadrature points is %d\n",q_points_all.size());
//     //control sum
//     #ifdef DEBUG    //----------------------
//     double sum = 0;
//     for(unsigned int i = 0; i < q_points_all.size(); i++)
//     {
//         sum += jxw_all[i];
//     }
//     sum = std::abs(sum-1.0);
//     if(sum > 1e-15) DBGMSG("Control sum of weights: %e\n",sum);
//     MASSERT(sum < 1e-12, "Sum of weights of quadrature points must be 1.0.\n");
//     #endif          //----------------------
}



void Adaptive_integration::gnuplot_refinement(const std::string &output_dir, bool real, bool show)
{
  //DBGMSG("gnuplotting\n");
  gather_w_points();    //gathers all quadrature points from squares into one vector and maps them to unit cell
  // print only adaptively refined elements
  if(q_points_all.size() <= quadratures_[6].size()) return;
  DBGMSG("level = %d,  number of quadrature points = %d\n",level_,q_points_all.size());
  
  std::string fgnuplot_ref = "adaptive_integration_refinement_",
              fgnuplot_qpoints = "adaptive_integration_qpoints_",
              script_file = "g_script_adapt_",
              felements = "elements";
  
              fgnuplot_ref += std::to_string(cell->index()) + ".dat";
              fgnuplot_qpoints += std::to_string(cell->index()) + ".dat";
              script_file += std::to_string(cell->index()) + ".p";
  try
    {
        Gnuplot g1("adaptive_integration");
        //g1.savetops("test_output");
        //g1.set_title("adaptive_integration\nrefinement");
        //g1.set_grid();
        
        std::ofstream felements_file;
        felements_file.open (output_dir + felements, ios_base::app);
        if (felements_file.is_open()) 
        {
            //reordering
            felements_file << cell->vertex(0) << "\n"
                << cell->vertex(1) << "\n"
                << cell->vertex(3) << "\n"
                << cell->vertex(2) << "\n"
                << cell->vertex(0) << "\n\n";
        }
        else 
        { 
          std::cout << "Coud not write refinement for gnuplot.\n";
        }
        felements_file.close();
        
        std::ofstream myfile1;
        myfile1.open (output_dir + fgnuplot_ref);
        if (myfile1.is_open()) 
        {
       
        for (unsigned int i = 0; i < squares.size(); i++)
        {
          if(real) squares[i].transform_to_real_space(cell, *mapping);
          for (unsigned int j = 0; j < 4; j++) 
          {
            if(real)
                myfile1 << squares[i].real_vertex(j);
            else
              myfile1 << squares[i].vertex(j);
            
            myfile1 << "\n";
          }
          if(real)
              myfile1 << squares[i].real_vertex(0);
            else
              myfile1 << squares[i].vertex(0);

          myfile1 << "\n\n";
        }
        std::cout << left << setw(53) <<  "Adaptive XFEM element refinement written in: " << fgnuplot_ref << endl;
        }
        else 
        { 
          std::cout << "Coud not write refinement for gnuplot.\n";
        }
        myfile1.close();
        
        std::ofstream myfile2;
        myfile2.open (output_dir + fgnuplot_qpoints);
        if (myfile2.is_open()) 
        {
       
            for (unsigned int q = 0; q < q_points_all.size(); q++)
            {
                if(real)
                    myfile2 << mapping->transform_unit_to_real_cell(cell, q_points_all[q]);
                else
                    myfile2 << q_points_all[q];
            myfile2 << "\n";
            }
        
            std::cout << left << setw(53) <<  "Quadrature points written in: " << fgnuplot_qpoints << std::endl;
        }
        else 
        { 
          std::cout << "Coud not write qpoints for gnuplot." << std::endl;
        }
        myfile2.close();
        
        //g1.set_style("lines").plotfile_xy("adaptive_integration_refinement.dat",1,2,"Adaptive integration refinement");
        
        //g1.set_multiplot();
        
        //creating command
        std::ostringstream strs;
        strs << "set terminal x11\n";
        strs << "set size ratio -1\n";
        strs << "set parametric\n";
        strs << "set trange [0:2*pi]\n";
        
        //unsigned int w = 0;
        for(unsigned int w = 0; w < xdata->n_wells(); w++)
        {
          /* # parametricly plotted circle
           * set parametric
           * set trange [0:2*pi]
           * # Parametric functions for a circle
           * fx(t) = r*cos(t)
           * fy(t) = r*sin(t)
           * plot fx(t),fy(t)
           */
          Well* well = xdata->get_well(w);
          if(real)
          {
            strs << "fx" << w << "(t) = " << well->center()[0] 
                << " + "<< well->radius() << "*cos(t)\n";
            strs << "fy" << w << "(t) = " << well->center()[1] 
                << " + "<< well->radius() << "*sin(t)\n";
          }
          else
          {
            Point<2> real_well_center = mapping->transform_real_to_unit_cell(cell, well->center());
                    
            strs << "fx" << w << "(t) = " << real_well_center[0] 
                << " + "<< well->radius() << "*cos(t)\n";
            strs << "fy" << w << "(t) = " << real_well_center[1] 
                << " + "<< well->radius() << "*sin(t)\n";
          }
        }
        
        strs << "plot \"" << fgnuplot_ref << "\" using 1:2 with lines,\\\n"
             << "\"" << fgnuplot_qpoints << "\" using 1:2 with points lc rgb \"light-blue\",\\\n";
        for(unsigned int w = 0; w < xdata->n_wells(); w++)
        {
          strs << "fx" << w << "(t),fy" << w << "(t)";
          if(w != xdata->n_wells()-1) 
            strs << ",\\\n";
        } 
        
        //saving gnuplot script
        std::ofstream myfile3;
        myfile3.open (output_dir + script_file);
        if (myfile3.is_open()) 
        {
          // header
          myfile3 << "# Gnuplot script for printing adaptively refined element.\n" <<
                     "# Made by Pavel Exner.\n#\n" <<
                     "# Run the script in gnuplot:\n" <<
                     "# > load \"" << script_file << "\"\n#\n" <<
                     "# Data files used:\n" << 
                     "# " << fgnuplot_ref << "\n"
                     "# " << fgnuplot_qpoints << "\n" 
                     "#\n#" << std::endl;
          // script
          myfile3 << strs.str() << std::endl;
          
          std::cout << left << setw(53) << "Gnuplot script for adaptive refinement written in: " << script_file << endl;
        }
        else 
        { 
          std::cout << "Coud not write gnuplot script.\n";
        }
        myfile3.close();
        
        
        //show the plot by gnuplot if show == true
        if(show)
        {
          //finally plot
          g1.cmd(strs.str());
        
          //g1.unset_multiplot();
          
          //g1.set_style("points").plot_xy(x,y,"user-defined points 2d");
        
          //g1.showonscreen(); // window output
         
          #if defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
          std::cout << std::endl << "GNUPLOT output on cell " << cell->index() << " ... Press ENTER to continue..." << std::endl;

          std::cin.clear();
          std::cin.ignore(std::cin.rdbuf()->in_avail());
          std::cin.get();
          #endif
        }
        
    return;
    
    }
    catch (GnuplotException ge)
    {
      std::cout << ge.what() << std::endl;
    }
}



double Adaptive_integration::test_integration(Function< 2 >* func)
{
    DBGMSG("integrating...\n");
    double integral = 0;
    gather_w_points();    //gathers all quadrature points from squares into one vector and maps them to unit cell
    if (q_points_all.size() == 0) return 0;
    
    Quadrature<2> quad(q_points_all, jxw_all);
    XFEValues<Enrichment_method::xfem_shift> xfevalues(*fe,quad, 
                                        update_quadrature_points | update_JxW_values );
    xfevalues.reinit(xdata);
  
    for(unsigned int q=0; q<q_points_all.size(); q++)
    {
//         if(q % 500 == 0) DBGMSG("q: %d\n",q);
        integral += func->value(xfevalues.quadrature_point(q)) * xfevalues.JxW(q);
    }
//     DBGMSG("q done\n");
    return integral;
}


std::pair< double, double > Adaptive_integration::test_integration_2(Function< 2 >* func, unsigned int diff_levels)
{   
    QGauss<2> gauss_quad = Adaptive_integration::quadratures_[1];
    
    // Prepare quadrature points - only on squares on the well edge
    MASSERT(squares.size() > 1, "Element not refined.");
    q_points_all.reserve(squares.size()*gauss_quad.size());
    jxw_all.reserve(squares.size()*gauss_quad.size());

    for(unsigned int i = 0; i < squares.size(); i++)
    {   
        //TODO: this will not include the squares with cross-section with well and no nodes inside
        //all the squares on the edge of the well
        bool on_well_edge = false;
        for(unsigned int w=0; w < xdata->n_wells(); w++)
        {
            unsigned int n = refine_criterion_nodes_in_well(squares[i],*(xdata->get_well(w)));
            if( (n > 0) && (n < 4)) on_well_edge = true;
        }
        if(on_well_edge)    
        {
            squares[i].gauss = &(gauss_quad);
            std::vector<Point<2> > temp(squares[i].gauss->get_points());
            squares[i].mapping.map_unit_to_real(temp);  //mapped from unit square to unit cell

            for(unsigned int w=0; w < xdata->n_wells(); w++)
            for(unsigned int j = 0; j < temp.size(); j++)
            {
                //include only points outside the well
                Point<2> real_quad = mapping->transform_unit_to_real_cell(cell, temp[j]);
                if(xdata->get_well(w)->center().distance(real_quad) >= xdata->get_well(w)->radius())
                {
                q_points_all.push_back(temp[j]);
                jxw_all.push_back( squares[i].gauss->weight(j) *
                                   squares[i].mapping.jakobian() );
                }
            }
            squares[i].refine_flag = false;
        }
        //all the squares around the well
        else
        {
            squares[i].refine_flag = true;
            //nothing - we ignore other squares
        }
    }
    q_points_all.shrink_to_fit();
    jxw_all.shrink_to_fit();
    
//     gnuplot_refinement("../output/test_adaptive_integration_2/",true);
//     squares.erase(std::remove_if(squares.begin(), squares.end(),remove_square_cond), squares.end());
    for (auto it = squares.begin(); it != squares.end(); ) {
        if (it->refine_flag)
            // new erase() that returns iter..
            it = squares.erase(it);
        else
            ++it;
    }
 
//     gnuplot_refinement("../output/test_adaptive_integration_2/",true);
    
    DBGMSG("squares refined and prepared...\n");
    
    Quadrature<2>* quad = new Quadrature<2>(q_points_all, jxw_all);
    XFEValues<Enrichment_method::xfem_shift> *xfevalues = 
        new XFEValues<Enrichment_method::xfem_shift>(*fe,*quad, update_quadrature_points | update_JxW_values );
    xfevalues->reinit(xdata);
  
    double integral_test = 0;
    for(unsigned int q=0; q<q_points_all.size(); q++)
    {
        integral_test += func->value(xfevalues->quadrature_point(q)) * xfevalues->JxW(q);
    }

    DBGMSG("test integral computed...%f\n",integral_test);
    
    //refine several times
    for(unsigned int i = 0; i < diff_levels; i++)
    {
        unsigned int squares_to_refine = squares.size();
        DBGMSG("refining...\n");
        for(unsigned int i = 0; i < squares.size(); i++)
            squares[i].refine_flag = true;
        refine(squares_to_refine);
    }
        
    //gnuplot_refinement("../output/test_adaptive_integration_2/",true);
    // Prepare quadrature points - only on squares on the well edge
    MASSERT(squares.size() > 1, "Element not refined.");
    q_points_all.clear();
    jxw_all.clear();
    q_points_all.reserve(squares.size()*quadratures_[1].size());
    jxw_all.reserve(squares.size()*quadratures_[1].size());

    for(unsigned int i = 0; i < squares.size(); i++)
    {   
            squares[i].gauss = &(gauss_quad);
            std::vector<Point<2> > temp(squares[i].gauss->get_points());
            squares[i].mapping.map_unit_to_real(temp);  //mapped from unit square to unit cell

            for(unsigned int w=0; w < xdata->n_wells(); w++)
            for(unsigned int j = 0; j < temp.size(); j++)
            {
                //include only points outside the well
                Point<2> real_quad = mapping->transform_unit_to_real_cell(cell, temp[j]);
                if(xdata->get_well(w)->center().distance(real_quad) >= xdata->get_well(w)->radius())
                {
                q_points_all.push_back(temp[j]);
//                 DBGMSG("i=%d \t w=%f \t j=%f\n",i,squares[i].gauss->weight(j), squares[i].mapping.jakobian());
                jxw_all.push_back( squares[i].gauss->weight(j) *
                                   squares[i].mapping.jakobian() );
                }
            }
    }
    q_points_all.shrink_to_fit();
    jxw_all.shrink_to_fit();
    
    //gnuplot_refinement("../output/test_adaptive_integration_2/",true);
    DBGMSG("squares refined and prepared...\n");
    
    delete xfevalues;
    delete quad;
    
    quad = new Quadrature<2>(q_points_all, jxw_all);
    xfevalues = 
        new XFEValues<Enrichment_method::xfem_shift>(*fe,*quad, update_quadrature_points | update_JxW_values );
    xfevalues->reinit(xdata);
    double integral_fine = 0;
    for(unsigned int q=0; q<q_points_all.size(); q++)
    {
//         std::cout << xfevalues->quadrature_point(q) << std::endl;
//         DBGMSG("q=%d \t jxw=%f\n",q,xfevalues->JxW(q));
        integral_fine += func->value(xfevalues->quadrature_point(q)) * xfevalues->JxW(q);
    }
//     cout << jxw_all[0] << "  " << q_points_all[10] <<endl;
    DBGMSG("fine integral computed...%f\n",integral_fine);
    
    delete xfevalues;
    delete quad;
    
    return std::make_pair(integral_test, integral_fine);
}