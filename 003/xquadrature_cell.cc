
#include "xquadrature_cell.hh"
#include "system.hh"
#include "data_cell.hh"
#include "mapping.hh"
#include "well.hh"
#include "gnuplot_i.hpp"

const std::vector<double> XQuadratureCell::alpha_ = 
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
const double XQuadratureCell::c_empiric_ = 12.65;
const double XQuadratureCell::p_empiric_ = 2.27;


XQuadratureCell::XQuadratureCell(XDataCell* xdata, 
                                 const Mapping< 2 >& mapping,
                                 Refinement::Type type
                                )
  : XQuadratureBase(),
    xdata_(xdata),
    mapping_(&mapping),
    refinement_type_(type)
{
    MASSERT(xdata_ != nullptr, "XDataCell in xquadrature is a null pointer."); 
    //first square
    squares_.push_back(Square(Point<2>(0,0), Point<2>(1,1)));
    
    for(unsigned int w = 0; w < xdata_->n_wells(); w++)
    {
      //if the whole square is inside the well
      if(refine_criterion_nodes_in_well(squares_[0],*(xdata_->get_well(w))) == 4)
      {
          squares_[0].gauss = &(XQuadratureBase::quadratures_[0]);
          squares_[0].processed = true;
      }
    }
}

void XQuadratureCell::refine(unsigned int max_level)
{
    //DBGMSG("cell: %d .................callling adaptive_integration.........\n",cell->index());
    
    for(unsigned int t=0; t < max_level; t++)
    {
        switch (refinement_type_)
        {
            case Refinement::edge:
                if (refine_edge()) continue;
             
            case Refinement::error:
                if (refine_error(alpha_tolerance_)) continue;
             
            case Refinement::polar:
                if (refine_polar()) continue;
        }
        break;  // if not continuing with refinement - break for cycle
    }
    
    gather_weights_points();
    map_quadrature_points_to_real();
    
    for(auto &sq: squares_)
        sq.transform_to_real_space(xdata_->get_cell(), *mapping_);
}


void XQuadratureCell::map_quadrature_points_to_real()
{
    real_points_.clear();
    real_points_.reserve(quadrature_points.size());
    
    for(unsigned int q = 0; q < quadrature_points.size(); q++)
    {
        // get rid of the quadrature points inside a well
        bool include_point=true;
        Point<2> real_quad = mapping_->transform_unit_to_real_cell(xdata_->get_cell(), quadrature_points[q]);
        for(unsigned int w=0; w < xdata_->n_wells(); w++)
        {
            //include only points outside the well
            if(xdata_->get_well(w)->center().distance(real_quad) <= xdata_->get_well(w)->radius())
                include_point = false;
        }
        if(include_point)
        {
            real_points_.push_back(real_quad);//mapping_->transform_unit_to_real_cell(cell_,quadrature_points[q]);
        }
        else
        {
            weights.erase(weights.begin()+q);
            quadrature_points.erase(quadrature_points.begin()+q);
            q--;
        }
    }
    real_points_.shrink_to_fit();
}

bool XQuadratureCell::refine_criterion_a(Square& square, Well& well)
{
    //return false; // switch on and off the criterion
    square.transform_to_real_space(xdata_->get_cell(), *mapping_);
    
    double min_distance = square.real_vertex(0).distance(well.center());// - well.radius();
    for(unsigned int j=1; j < 4; j++)
    {
        double dist = well.center().distance(square.real_vertex(j));// - well.radius();
        min_distance = std::min(min_distance,dist);
    }

    //DBGMSG("square [%d] diameter=%f , min_distance=%f cell_diameter=%f\n",i,squares_[i].real_diameter(),min_distance, cell->diameter());
    // criteria:
    if( square.real_diameter() > square_refinement_criteria_factor_ * min_distance)
        return true;
    else return false;
}

unsigned int XQuadratureCell::refine_criterion_nodes_in_well(Square& square, Well& well)
{
    square.transform_to_real_space(xdata_->get_cell(), *mapping_);

    unsigned int vertices_in_well = 0;
    for(unsigned int j=0; j < 4; j++)
    {
        if(well.points_inside(square.real_vertex(j)))
            vertices_in_well++;
    }
    //DBGMSG("nodes in well %d\n",vertices_in_well);
    return vertices_in_well;
}

unsigned int XQuadratureCell::refine_criterion_alpha(double r_min)
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

bool XQuadratureCell::refine_criterion_h(Square& square, Well& well, double criterion_rhs)
{   
    //DBGMSG("h criterion: h=%e, rhs=%e\n",square.real_diameter(), criterion_rhs);
    return (square.real_diameter() > criterion_rhs);
}

// bool XQuadratureBase::refine_criterion_r_min(Square& square, double r_min)
// {
//     //return false; // switch on and off the criterion
//     square.transform_to_real_space(cell, *mapping);
//     
//     if( square.real_diameter() > (square_refinement_criteria_factor_ * r_min) )
//         return true;
//     else return false;
// }

double XQuadratureCell::compute_r_min(Square& square, unsigned int w)
{
    Well well = *(xdata_->get_well(w));
    Point<2> wc = well.center();
    std::vector<unsigned int> quadrants(4); //quadrants are numbered in standard counterclockwise sence
    
    // determine which square vertices lies in which quadrant of the well center
    for(unsigned int j = 0; j < 4; j++)
    {
        int x = (square.real_vertex(j)(0)-wc(0)) >= 0;
        int y = (square.real_vertex(j)(1)-wc(1)) >= 0;
        //http://www.codeproject.com/Questions/47032/find-quadrant-of-a-point-on-graph-without-if-condi
        //http://stackoverflow.com/questions/9718059/determining-the-quadrant-of-a-point
        quadrants[j] =  3 + x - y - 2 * x * y;         
    }
//     DBGMSG("quadrants: %d%d%d%d\n",quadrants[0], quadrants[1], quadrants[2], quadrants[3]);
    Tensor<1,2> direction_vector;
    unsigned int a,b;
    double lines_parameter;
    Point<2> t_point;
    if( (quadrants[0] != quadrants[1])) //vertex 0 and 1 does not lie in same quadrant
    {
        a = quadrants[1]-1;
        b = (a == 0) ? 1 : 2;
    }
    else if( (quadrants[1] != quadrants[2])) //vertex 1 and 2 does not lie in same quadrant
    {
        a = quadrants[2]-1;
        b = (a == 0) ? 3 : 2;
    }
    else    // all points are in the same quadrants
    {
        //quadrant corresponds with the closest vertex (draw a picture)
        return square.real_vertex(quadrants[0]-1).distance(well.center());
    }
    
    //find point T which lies on line [a,b] and is closest to well center
    direction_vector = square.real_vertex(b) - square.real_vertex(a);
    lines_parameter = ( direction_vector * 
                                (wc - square.real_vertex(a))
                            ) / direction_vector.norm_square();
                            
    t_point = lines_parameter*direction_vector + square.real_vertex(a);
    return well.center().distance(t_point);
}


bool XQuadratureCell::refine_edge()
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
  
    for(unsigned int i = 0; i < squares_.size(); i++)
    { 
        if(squares_[i].processed) 
        {
            //DBGMSG("processed\n");
            continue;
        }
                
        for(unsigned int w = 0; w < xdata_->n_wells(); w++)
        {
            well = xdata_->get_well(w);
            //skip squares_ that are already flagged
            if(squares_[i].refine_flag) continue;
        
            // refinement on the cells without well inside
            if( xdata_->q_points(w).size() == 0)    // is the well not inside ? )
            {
                squares_[i].gauss = &(XQuadratureBase::quadratures_[3]);
                //minimum distance from well criterion      ------------------------------------[5]
                if( refine_criterion_a(squares_[i],*well) )
                {
                    squares_[i].refine_flag = true;
                    n_squares_to_refine++;
                }
            }
            else
            {
                //testing the edge of the well
                //by the distance of a point (center of the well) to a line (side of the square)
      
                n_nodes_in_well = refine_criterion_nodes_in_well(squares_[i],*well);

                //if the whole square is not inside the the well      ------------------------------------[1]
                if(n_nodes_in_well !=0 && n_nodes_in_well < 4)
                {
                    //squares on the edge of the well obtain three point quadrature
                    squares_[i].gauss = &(XQuadratureBase::quadratures_[3]);
                    //std::cout << i << " addded(node)\n";
                    n_squares_to_refine++;
                    squares_[i].refine_flag = true;
                    continue;
                }
                //if the whole square is inside the well              ------------------------------------[2]
                if (n_nodes_in_well == 4) 
                {
                    //squares inside the well obtain one point quadrature
                    squares_[i].gauss = &(XQuadratureBase::quadratures_[0]);
                    continue;
                }
      
                // temporary shortcuts
                const Point<2>* vertices = squares_[i].real_vertices();
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
                    squares_[i].gauss = &(XQuadratureBase::quadratures_[3]);
                    n_squares_to_refine++;
                    squares_[i].refine_flag = true;
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
                        squares_[i].gauss = &(XQuadratureBase::quadratures_[3]);
                        //std::cout << i << " addded(line)\n";
                        n_squares_to_refine++;
                        squares_[i].refine_flag = true;
                        break;
                    }
                    }
                }

                if(squares_[i].refine_flag) continue;
                
                //minimum distance from well criterion      ------------------------------------[5]
                if( refine_criterion_a(squares_[i],*well) )
                {
                        squares_[i].gauss = &(XQuadratureBase::quadratures_[3]);
                        squares_[i].refine_flag = true;
                        n_squares_to_refine++;
                }
                //std::cout << std::endl;
            }   // if
        } // for w
        
        if(! squares_[i].refine_flag) squares_[i].processed = true;
    } // for i
  
  if (n_squares_to_refine == 0) 
    return false;
  else
  {
    apply_refinement(n_squares_to_refine);
    return true;
  }
}



bool XQuadratureCell::refine_error(double alpha_tolerance)
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
  
    for(unsigned int i = 0; i < squares_.size(); i++)
    { 
        //DBGMSG("square[%d]:\n",i);
        if(squares_[i].processed) 
        {
            //DBGMSG("processed\n");
            continue;
        }
                
        for(unsigned int w = 0; w < xdata_->n_wells(); w++)
        {
            well = *(xdata_->get_well(w));
            h_criterion_rhs = std::pow(alpha_tolerance_ / c_empiric_ * std::pow(well.radius(),p_empiric_), 
                                       1.0/(p_empiric_-1.0)); 
            //skip squares that are already flagged
            if(squares_[i].refine_flag) continue;
        
            // refinement on the cells without well inside
//             if( xdata->q_points(w).size() == 0)    // is the well not inside ? )
//             {
//                 //minimum distance from well criterion      ------------------------------------[5]
//                 if( refine_criterion_a(squares_[i],*well) )
//                 {
//                     squares_[i].refine_flag = true;
//                     n_squares_to_refine++;
//                 }
//             }
//             else
            {
                //testing the edge of the well
                //by the distance of a point (center of the well) to a line (side of the square)
      
                n_nodes_in_well = refine_criterion_nodes_in_well(squares_[i], well);

                //if the whole square is inside the well              ------------------------------------[1]
                if (n_nodes_in_well == 4) 
                {
                    //squares inside the well obtain no quadrature points
                    //DBGMSG("square[%d]: inside well\n", i);
                    squares_[i].gauss = &(XQuadratureBase::quadratures_[0]);
                    continue;
                }
                
                //if the whole square is not inside the the well and criterion h -------------------------[2]
                if(n_nodes_in_well !=0 && n_nodes_in_well < 4)
                {
                    //DBGMSG("square[%d]: node in well\n",i);
                    if(refine_criterion_h(squares_[i], well, h_criterion_rhs))
                    {
                        n_squares_to_refine++;
                        squares_[i].refine_flag = true;
                    }
                    else
                    {
                        squares_[i].gauss = &(XQuadratureBase::quadratures_[1]);
                        //squares[i].refine_flag = false;
                    }
                    continue;
                }

      
                //if the whole well is inside the square              ------------------------------------[3]
                if ( n_nodes_in_well == 0 
                    && (well.center()[0] >= squares_[i].real_vertex(0)[0]) 
                    && (well.center()[0] <= squares_[i].real_vertex(2)[0])
                    && (well.center()[1] >= squares_[i].real_vertex(0)[1]) 
                    && (well.center()[1] <= squares_[i].real_vertex(2)[1]) 
                    ) 
                {
                    //DBGMSG("square[%d]: well inside the square\n", i);
                    //squares outside the well obtain three point quadrature
                    //squares[i].gauss = &(XQuadratureBase::gauss_3);
                    n_squares_to_refine++;
                    squares_[i].refine_flag = true;
                    continue;
                }
                
                double r_min = compute_r_min(squares_[i], w);
                //DBGMSG("r_min = %e\n",r_min);
                double val = squares_[i].real_diameter() / r_min;
                if(val > square_refinement_criteria_factor_)
                {
                    //DBGMSG("square[%d]: refine: h/rmin = %d\n",i, val);
                    n_squares_to_refine++;
                    squares_[i].refine_flag = true;
                    continue;
                }
                else
                {
                    unsigned int quad_order = refine_criterion_alpha(r_min);
                    if(quad_order < XQuadratureBase::quadratures_.size())
                    {
                        //DBGMSG("square[%d]:\n",i);
                        if (quad_order == 1) quad_order = 2;
                        squares_[i].gauss = &(XQuadratureBase::quadratures_[quad_order]);
                    }
                    else
                    {
                        //DBGMSG("square[%d]: refine: alpha criterion failed\n", i);
                        n_squares_to_refine++;
                        squares_[i].refine_flag = true;
                    }
                    continue;
                }
                
            }   // if
        } // for w
        
        if(! squares_[i].refine_flag) squares_[i].processed = true;
    } // for i
  
  if (n_squares_to_refine == 0) 
    return false;
  else
  {
    apply_refinement(n_squares_to_refine);
    return true;
  }
}


bool XQuadratureCell::refine_polar()
{
    unsigned int n_squares_to_refine = 0; 
    unsigned int n_nodes_in_well;
    Well* well;
  /* there are several cases that can happen:
   * 1] if all nodes of a square are inside a well 
   *    the whole square is inside                    -> no refine
   * 2] r_min > well_radius  &  square_diameter > C*r_min       -> refine
   * 3] r_min > well_radius  &  square_diameter < C*r_min       -> 3x3 Gauss, no refine
   * 
   * 4] r_min < well_radius  &  square_diameter > well_radius       -> refine
   *    if we lower tolerance here, we obtain similar behaviour as with 2^(-10) stop stopping rule
   * 
   * 5] r_min < well_radius  &  square_diameter < well_radius       -> 3x3 Gauss, no refine
   *
   * 6] if the minimum distance of square vertex 
   *    to well edge is smaller than square diameter  -> refine
   */
  //DBGMSG("wells.size(): %d", xdata->wells().size());
  
    for(unsigned int i = 0; i < squares_.size(); i++)
    { 
        if(squares_[i].processed) 
        {
            //DBGMSG("processed\n");
            continue;
        }
                
        for(unsigned int w = 0; w < xdata_->n_wells(); w++)
        {
            well = xdata_->get_well(w);
            //skip squares that are already flagged
            if(squares_[i].refine_flag) continue;
        
            // refinement on the cells without well inside
            if( xdata_->q_points(w).size() == 0)    // is the well not inside ? )
            {
                squares_[i].gauss = &(XQuadratureBase::quadratures_[3]);
                //minimum distance from well criterion      ------------------------------------[6]
                if( refine_criterion_a(squares_[i],*well) )
                {
                    squares_[i].refine_flag = true;
                    n_squares_to_refine++;
                }
            }
            else
            {
                //testing the edge of the well
                //by the distance of a point (center of the well) to a line (side of the square)
      
                n_nodes_in_well = refine_criterion_nodes_in_well(squares_[i],*well);

                //if the whole square is inside the well              ------------------------------------[1]
                if (n_nodes_in_well == 4) 
                {
                    //squares inside the well obtain one point quadrature
                    squares_[i].gauss = &(XQuadratureBase::quadratures_[0]);
                    continue;
                }
                
                double r_min = compute_r_min(squares_[i],w);
//                 DBGMSG("square %d, r_min = %f\n",i, r_min);
                if(r_min > well->radius())
                {
                    squares_[i].gauss = &(XQuadratureBase::quadratures_[3]);
                    //if square size is larger than C*r_min               ---------------------------------[2]
                    if (squares_[i].real_diameter() > (square_refinement_criteria_factor_ * r_min))
                    {
                        n_squares_to_refine++;
                        squares_[i].refine_flag = true;
//                         DBGMSG("A refine square %d\n",i);
                    }
                    // else square size is smaller than C*r_min - no refine -------------------------------[3]
                    continue;
                }
                else
                {
                    squares_[i].gauss = &(XQuadratureBase::quadratures_[3]);
                    //if square size is larger than C*r_min               ---------------------------------[4]
                    if (squares_[i].real_diameter() > (square_refinement_criteria_factor_ *well->radius()))
                    {
                        n_squares_to_refine++;
                        squares_[i].refine_flag = true;
//                         DBGMSG("B refine square %d\n",i);
                        
                    }
                    // else square size is smaller than well_radius - no refine ---------------------------[5]
                    continue;
                }
            }   // if
        } // for w
        
        if(! squares_[i].refine_flag) squares_[i].processed = true;
    } // for i
  
  if (n_squares_to_refine == 0) 
    return false;
  else
  {
    apply_refinement(n_squares_to_refine);
    return true;
  }
}


void XQuadratureCell::gnuplot_refinement(const string& output_dir, bool real, bool show)
{ 
  if(level_ < 1) return;
  DBGMSG("level = %d,  number of quadrature points = %d\n",level_, quadrature_points.size());
  
  DoFHandler<2>::active_cell_iterator cell =  xdata_->get_cell();   //shortcut
  
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
       
        for (unsigned int i = 0; i < squares_.size(); i++)
        {
          for (unsigned int j = 0; j < 4; j++) 
          {
            if(real)
                myfile1 << squares_[i].real_vertex(j);
            else
              myfile1 << squares_[i].vertex(j);
            
            myfile1 << "\n";
          }
          if(real)
              myfile1 << squares_[i].real_vertex(0);
            else
              myfile1 << squares_[i].vertex(0);

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
       
            for (unsigned int q = 0; q < quadrature_points.size(); q++)
            {
                if(real)
                    myfile2 << real_points_[q];
                else
                    myfile2 << quadrature_points[q];
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
        for(unsigned int w = 0; w < xdata_->n_wells(); w++)
        {
          /* # parametricly plotted circle
           * set parametric
           * set trange [0:2*pi]
           * # Parametric functions for a circle
           * fx(t) = r*cos(t)
           * fy(t) = r*sin(t)
           * plot fx(t),fy(t)
           */
          Well* well = xdata_->get_well(w);
          if(real)
          {
            strs << "fx" << w << "(t) = " << well->center()[0] 
                << " + "<< well->radius() << "*cos(t)\n";
            strs << "fy" << w << "(t) = " << well->center()[1] 
                << " + "<< well->radius() << "*sin(t)\n";
          }
          else
          {
              //disable mapping
//             Point<2> real_well_center = mapping->transform_real_to_unit_cell(cell, well->center());
                    
            strs << "fx" << w << "(t) = " << 0 // real_well_center[0] 
                << " + "<< well->radius() << "*cos(t)\n";
            strs << "fy" << w << "(t) = " << 0 // real_well_center[1] 
                << " + "<< well->radius() << "*sin(t)\n";
          }
        }
        
        strs << "plot \"" << fgnuplot_ref << "\" using 1:2 with lines,\\\n"
             << "\"" << fgnuplot_qpoints << "\" using 1:2 with points lc rgb \"light-blue\",\\\n";
        for(unsigned int w = 0; w < xdata_->n_wells(); w++)
        {
          strs << "fx" << w << "(t),fy" << w << "(t)";
          if(w != xdata_->n_wells()-1) 
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