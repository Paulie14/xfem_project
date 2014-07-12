
#include <string>

#include "adaptive_integration.hh"
#include "system.hh"
#include "gnuplot_i.hpp"

#include "data_cell.hh"
#include "mapping.hh"
#include "xfevalues.hh"

const double Adaptive_integration::square_refinement_criteria_factor = 1.0;
const QGauss<2> Adaptive_integration::gauss_1 = QGauss<2>(1);
const QGauss<2> Adaptive_integration::gauss_2 = QGauss<2>(2);
const QGauss<2> Adaptive_integration::gauss_3 = QGauss<2>(3);
const QGauss<2> Adaptive_integration::gauss_4 = QGauss<2>(3);


Square::Square(const Point< 2 > &p1, const Point< 2 > &p2)
  : refine_flag(false),
    gauss(&(Adaptive_integration::gauss_4)),
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
    vertices[0] = Point<2>(p2[0], p1[1]);
    vertices[1] = p1;
    vertices[2] = Point<2>(p1[0], p2[1]);
    vertices[3] = p2;
    mapping = MyMapping(vertices[0],vertices[2]);
    return;
  }
  
  //renumbering vertices - got 3 and 1
  if(p1[0] < p2[0] && p1[1] > p2[1])
  {
    vertices[0] = Point<2>(p1[0], p2[1]);
    vertices[1] = p2;
    vertices[2] = Point<2>(p2[0], p1[1]);
    vertices[3] = p1;
    mapping = MyMapping(vertices[0],vertices[2]);
    return;
  }
  
  //renumbering vertices - got 2 and 0
  if(p1[0] > p2[0])
  {
    vertices[0] = p2;
    vertices[1] = Point<2>(p1[0],p2[1]);
    vertices[2] = p1;
    vertices[3] = Point<2>(p2[0],p1[1]);
    mapping = MyMapping(vertices[0],vertices[2]);
    return;
  }
  
   //renumbering vertices - got 0 and 2
  vertices[0] = p1;
  vertices[1] = Point<2>(p2[0],p1[1]);
  vertices[2] = p2;
  vertices[3] = Point<2>(p1[0],p2[1]);
  
  unit_diameter_ = p1.distance(p2);
  mapping = MyMapping(vertices[0],vertices[2]);
}


void Square::transform_to_real_space(const DoFHandler< 2  >::active_cell_iterator& cell,
                                     const Mapping< 2 >& cell_mapping)
{
    if( ! transformed_to_real_) // if not already transformed
    {
        //mapping.print(cout);
        for(unsigned int i=0; i<4; i++)
        {
            //real_vertices_[i] = mapping.map_unit_to_real(vertices[i]);  //map to unit cell
            real_vertices_[i] = cell_mapping.transform_unit_to_real_cell(cell,vertices[i]); // map to real cell
        }
        real_diameter_ = std::max(real_vertices_[0].distance(real_vertices_[2]),
                                  real_vertices_[1].distance(real_vertices_[3]));
        //DBGMSG("unit_diameter_=%f, real_diameter_=%f\n", unit_diameter_, real_diameter_);
        transformed_to_real_ = true;
    }
}



Adaptive_integration::Adaptive_integration(const DoFHandler< 2  >::active_cell_iterator& cell, 
                                           const dealii::FE_Q< 2 >& fe,
                                           const Mapping<2>& mapping
                                          )
  : cell(cell), fe(&fe), mapping (&mapping),
    cell_mapping(cell->vertex(0), cell->vertex(3)),
    dirichlet_function(nullptr),
    rhs_function(nullptr),
    level(0)
{
      MASSERT(cell->user_pointer() != NULL, "NULL user_pointer in the cell"); 
      //A *a=static_cast<A*>(cell->user_pointer()); //from DEALII (TriaAccessor)
      xdata = static_cast<XDataCell*>( cell->user_pointer() );
      //xdata->initialize();
      
      //first square
      squares.push_back(Square(Point<2>(0,0), Point<2>(1,1)));
      
      //mapping well radius to unit measure and well center to unit cell
      m_well_center.resize(xdata->n_wells());
      m_well_radius.resize(xdata->n_wells());
      Tensor<1,2> well_radius;
      
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
//         
//         if( well_inside[w] )
        if( xdata->q_points(w).size() > 0)    // is the well inside ?
        {
            //m_well_center[w] = cell_mapping.map_real_to_unit(xdata->get_well(w)->center());
            //DBGMSG("Map well center:\n");
            m_well_center[w] = mapping.transform_real_to_unit_cell(cell, xdata->get_well(w)->center());
            //DBGMSG("%f %f\n", m_well_center[w][0], m_well_center[w][1]);
            well_radius[0] = xdata->get_well(w)->radius();
            well_radius = cell_mapping.scale_inverse(well_radius);
            m_well_radius[w] = well_radius[0];
            //DBGMSG("m_well_radius=%e\n",m_well_radius[w]);
        }
        //else DBGMSG("adaptive refinement on cell without well, index = %d\n",cell->index());

      }
      
      //DBGMSG("Printing cell mapping:\n");
      //cell_mapping.print(std::cout);
      
      //for(unsigned int i=0; i < 4; i++)
      //  std::cout << cell_mapping.map_real_to_unit(cell->vertex(i)) << "  ";
}

bool Adaptive_integration::refine_criterion_a(Square& square, Well& well)
{
    square.transform_to_real_space(cell, *mapping);
    double min_distance = square.real_vertex(0).distance(well.center()) - well.radius();
    for(unsigned int j=1; j < 4; j++)
    {
        double dist = well.center().distance(square.real_vertex(j)) - well.radius();
        min_distance = std::min(min_distance,dist);
    }
    //DBGMSG("square [%d] diameter=%f , min_distance=%f cell_diameter=%f\n",i,squares[i].real_diameter(),min_distance, cell->diameter());
    // criteria:
    if( square.real_diameter() > square_refinement_criteria_factor * min_distance)
    {
                //DBGMSG("Refine square[%d], diameter=%f. \n",i,squares[i].real_diameter());
                //squares[i].refine_flag = true;
                //n_squares_to_refine++;
        return true;
    }
}

bool Adaptive_integration::refine_edge()
{
  Tensor<1,2> direction_vectors[4];
  double lines_parameters[4];
  double distances[4];
  Point<2> t_points[4];
  unsigned int n_squares_to_refine = 0; 
  unsigned int n_nodes_in_well=0;
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
  for(unsigned int w = 0; w < xdata->n_wells(); w++)
  {
    well = xdata->get_well(w);
    // refinement on the cells without well inside
    if( xdata->q_points(w).size() == 0)    // is the well not inside ? )
    {
        for(unsigned int i = 0; i < squares.size(); i++)
        {
            //minimum distance from well criterion      ------------------------------------[5]
            if( refine_criterion_a(squares[i],*well) )
            {
                squares[i].refine_flag = true;
                n_squares_to_refine++;
            }
        }
    }
    //DBGMSG("well center: %f %f\tradius: %f", m_well_center[w][0], m_well_center[w][1], m_well_radius[w]);
    
    // refinement on the cells with well inside
    if( xdata->q_points(w).size() > 0)    // is the well not inside ? )
    for(unsigned int i = 0; i < squares.size(); i++)
    {
      //skip squares that are already flagged
      //if(squares[i].refine_flag) continue;
      
      //testing the edge of the well
      //by the distance of a point (center of the well) to a line (side of the square)
      
      n_nodes_in_well = 0;
      //if the node of the square lies in the well circle than it must be definitely refine
      for(unsigned int j=0; j < 4; j++)
      {
        //if(squares[i].vertices[j].distance(m_well_center[w]) <= m_well_radius[w])
        if(xdata->get_well(w)->points_inside(mapping->transform_unit_to_real_cell(cell, squares[i].vertices[j])))
        {
          n_nodes_in_well++;
        }
      }
      //if the whole square is not inside the the well      ------------------------------------[1]
      if(n_nodes_in_well !=0 && n_nodes_in_well < 4)
      {
        //squares on the edge of the well obtain three point quadrature
        squares[i].gauss = &(Adaptive_integration::gauss_3);
        //std::cout << i << " addded(node)\n";
        n_squares_to_refine++;
        squares[i].refine_flag = true;
        squares[i].on_well_edge = true;
        continue;
      }
      //if the whole square is inside the well              ------------------------------------[2]
      if (n_nodes_in_well == 4) 
      {
        //squares inside the well obtain one point quadrature
        squares[i].gauss = &(Adaptive_integration::gauss_1);
        continue;
      }
      
      //if the whole well is inside the square              ------------------------------------[3]
      if ( n_nodes_in_well == 0 
           && (m_well_center[w][0] >= squares[i].vertices[0][0]) 
           && (m_well_center[w][0] <= squares[i].vertices[2][0])
           && (m_well_center[w][1] >= squares[i].vertices[0][1]) 
           && (m_well_center[w][1] <= squares[i].vertices[2][1]) 
         ) 
      {
        //squares outside the well obtain three point quadrature
        squares[i].gauss = &(Adaptive_integration::gauss_3);
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
        direction_vectors[j] = squares[i].vertices[j+1] - squares[i].vertices[j];
        lines_parameters[j] = ( direction_vectors[j] * 
                                (m_well_center[w] - squares[i].vertices[j])
                              ) / direction_vectors[j].norm_square();
                              
        t_points[j] = lines_parameters[j]*direction_vectors[j] + squares[i].vertices[j];
        distances[j] = m_well_center[w].distance(t_points[j]);
      }
      direction_vectors[3] = squares[i].vertices[0] - squares[i].vertices[3];
      lines_parameters[3] = (direction_vectors[3] * (m_well_center[w] - squares[i].vertices[3])) / direction_vectors[3].norm_square();
      t_points[3] = lines_parameters[3]*direction_vectors[3] + squares[i].vertices[3];
      distances[3] = m_well_center[w].distance(t_points[3]);
      
      //std::cout << "distance\t";
      for(unsigned int j = 0; j < 4; j++)
      {
        //std::cout << distances[j] << " _ ";
        if(distances[j] <= m_well_radius[w])
        {
          int a = j-1,
              b = j+1;
          if (j == 0) 
            a = 3;
          if (j == 3) 
            b = 0;
          
          //then the well edge crosses the square line------------------------------------[4]
          if( std::abs(distances[a] + distances[b] - 
                          squares[i].vertices[0].distance(squares[i].vertices[1])) < 1e-13) 
          {
            //squares on the edge of the well obtain three point quadrature
            squares[i].gauss = &(Adaptive_integration::gauss_3);
            //std::cout << i << " addded(line)\n";
            n_squares_to_refine++;
            squares[i].refine_flag = true;
            squares[i].on_well_edge = true;
            break;
          }
        }
      }
      if(squares[i].refine_flag) continue;
      
      //minimum distance from well criterion      ------------------------------------[5]
      if( refine_criterion_a(squares[i],*well) )
      {
            squares[i].gauss = &(Adaptive_integration::gauss_3);
            squares[i].refine_flag = true;
            n_squares_to_refine++;
      }
      //std::cout << std::endl;
    } // for squares
  } // for wells
  
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
      center = (squares[i].vertices[0] + squares[i].vertices[2]) / 2;
      for(unsigned int j = 0; j < 4; j++)
      {
        squares.push_back(Square(squares[i].vertices[j],center));
        
        
        //checking if the whole new square lies in the well
        n_nodes_in_well = 0;
        //if the node of the square lies in the well circle than it must be definitely refine
        for(unsigned int w = 0; w < xdata->n_wells(); w++)
        {
          for(unsigned int u=0; u < 4; u++)
          {
            if(xdata->get_well(w)->points_inside(mapping->transform_unit_to_real_cell(cell, squares.back().vertices[u])))
            {
              n_nodes_in_well++;
            }
          }
        }
        if (n_nodes_in_well == 4)
        {
          //if the whole square lies in the well then only one point quadrature is needed
          squares.back().gauss = &(Adaptive_integration::gauss_1);
        }
        else
        {
          //else it gets the quadrature from the descendant square
          squares.back().gauss = squares[i].gauss;
        }
      }
    }
  }
  
  for(unsigned int i = 0; i < n_original_squares; i++)
  {
    if(squares[i].refine_flag)
    {
      squares.erase(squares.begin()+i);
      i--;  //one erased, so we must lower iterator
    }
  }
  
  level ++;
  
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
    
    q_points_all.reserve(squares.size()*gauss_3.size());
    jxw_all.reserve(squares.size()*gauss_3.size());

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
            //all the squares on the edge of the well
            if(squares[i].on_well_edge)
            {
                std::vector<Point<2> > temp(squares[i].gauss->get_points());
                //temp = squares[i].gauss->get_points(); 
                squares[i].mapping.map_unit_to_real(temp);  //mapped from unit square to unit cell

                for(unsigned int w=0; w < xdata->n_wells(); w++)
                for(unsigned int j = 0; j < temp.size(); j++)
                {
                    //include only points outside the well
                    //TODO: this will not work on non-square mesh
                    if(m_well_center[w].distance(temp[j]) >= m_well_radius[w])
                    {
                    q_points_all.push_back(temp[j]);
                    jxw_all.push_back( squares[i].gauss->weight(j) *
                                   squares[i].mapping.jakobian() );
                    }
                }
            }
            //all the squares around the well
            else if( squares[i].gauss->size() > 1 )
            {
                std::vector<Point<2> > temp(squares[i].gauss->get_points());
                //temp = squares[i].gauss->get_points(); 
                squares[i].mapping.map_unit_to_real(temp);  //mapped from unit square to unit cell
    
                for(unsigned int j = 0; j < temp.size(); j++)
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
  std::string fgnuplot_ref = "adaptive_integration_refinement_",
              fgnuplot_qpoints = "adaptive_integration_qpoints_",
              script_file = "g_script_adapt_";
  
              fgnuplot_ref += std::to_string(cell->index()) + ".dat";
              fgnuplot_qpoints += std::to_string(cell->index()) + ".dat";
              script_file += std::to_string(cell->index()) + ".p";
  try
    {
        Gnuplot g1("adaptive_integration");
        //g1.savetops("test_output");
        //g1.set_title("adaptive_integration\nrefinement");
        //g1.set_grid();
        
        /*
        std::vector<double> x(squares.size()), y(squares.size());

        for (unsigned int i = 0; i < squares.size(); i++)
        {
          for (unsigned int j = 0; j < 4; j++) 
          {
            x.push_back(squares[i].vertices[j][0]);          
            y.push_back(squares[i].vertices[j][1]);
          }
        }
        */
        
        
        std::ofstream myfile1;
        myfile1.open (output_dir + fgnuplot_ref);
        if (myfile1.is_open()) 
        {
       
        for (unsigned int i = 0; i < squares.size(); i++)
        {
          for (unsigned int j = 0; j < 4; j++) 
          {
            if(real)
              myfile1 << mapping->transform_unit_to_real_cell(cell, squares[i].vertices[j]);
              //myfile1 << cell_mapping.map_unit_to_real(squares[i].vertices[j]);
            else
              myfile1 << squares[i].vertices[j];
            
            myfile1 << "\n";
          }
          if(real)
              myfile1 << mapping->transform_unit_to_real_cell(cell, squares[i].vertices[0]);
              //myfile1 << cell_mapping.map_unit_to_real(squares[i].vertices[j]);
            else
              myfile1 << squares[i].vertices[0];

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
        
//         for (unsigned int i = 0; i < squares.size(); i++)
//         {
//           for (unsigned int j = 0; j < squares[i].gauss->get_points().size(); j++) 
//           {
//             if(real)
//               myfile2 << mapping->transform_unit_to_real_cell(cell, squares[i].mapping.map_unit_to_real(squares[i].gauss->get_points()[j]));
//               //myfile2 << cell_mapping.map_unit_to_real(squares[i].mapping.map_unit_to_real(squares[i].gauss->get_points()[j]));
//             else
//               myfile2 << squares[i].mapping.map_unit_to_real(squares[i].gauss->get_points()[j]);
//              
//             myfile2 << "\n";
//           }
//         }
        
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
          if(real)
          {
            strs << "fx" << w << "(t) = " << xdata->get_well(w)->center()[0] 
                << " + "<< xdata->get_well(w)->radius() << "*cos(t)\n";
            strs << "fy" << w << "(t) = " << xdata->get_well(w)->center()[1] 
                << " + "<< xdata->get_well(w)->radius() << "*sin(t)\n";
          }
          else
          {
            strs << "fx" << w << "(t) = " << m_well_center[w][0] 
                << " + "<< m_well_radius[w] << "*cos(t)\n";
            strs << "fy" << w << "(t) = " << m_well_center[w][1] 
                << " + "<< m_well_radius[w] << "*sin(t)\n";
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
    double integral = 0;
    gather_w_points();    //gathers all quadrature points from squares into one vector and maps them to unit cell
    Quadrature<2> quad(q_points_all, jxw_all);
    XFEValues<Enrichment_method::xfem_shift> xfevalues(*fe,quad, update_values 
                                       | update_gradients 
                                       | update_quadrature_points 
                                       //| update_covariant_transformation 
                                       //| update_transformation_values 
                                       //| update_transformation_gradients
                                       //| update_boundary_forms 
                                       //| update_cell_normal_vectors 
                                       | update_JxW_values 
                                       //| update_normal_vectors
                                       //| update_contravariant_transformation
                                       //| update_q_points
                                       //| update_support_points
                                       //| update_support_jacobians 
                                       //| update_support_inverse_jacobians
                                       //| update_second_derivatives
                                       //| update_hessians
                                       //| update_volume_elements
                                       //| update_jacobians
                                       //| update_jacobian_grads
                                       //| update_inverse_jacobians
                                                 );
    xfevalues.reinit(xdata);
  
    for(unsigned int q=0; q<q_points_all.size(); q++)
    {
        integral += func->value(xfevalues.quadrature_point(q)) * xfevalues.JxW(q);
    }
    return integral;
}



//OBSOLETE
/*
////////////////////////////////////////////////////// INTEGRATE_XFEM_RAMP ////////////////////////////////////////////
void Adaptive_integration::integrate_xfem( FullMatrix<double> &cell_matrix, 
                                      Vector<double> &cell_rhs,
                                      std::vector<unsigned int> &local_dof_indices,
                                      const double &transmisivity)
{  
  unsigned int n_wells_inside = 0,          // number of wells with q_points inside the cell
               n_wells = xdata->n_wells(),  //number of wells affecting the cell
               dofs_per_cell = fe->dofs_per_cell,
               n_dofs = n_wells*dofs_per_cell + dofs_per_cell;
    
  //getting unenriched local dofs indices : [FEM(dofs_per_cell), XFEM(n_wells*dofs_per_cell), WELL(n_wells)]
  local_dof_indices.clear();
  local_dof_indices.resize(dofs_per_cell);
  cell->get_dof_indices(local_dof_indices);
  
  local_dof_indices.resize(n_dofs);
  //getting enriched dof indices and well indices
  for(unsigned int w = 0; w < n_wells; w++)
  {   
    for(unsigned int i = 0; i < dofs_per_cell; i++)
    {
      local_dof_indices[(w+1)*dofs_per_cell+i] = xdata->global_enriched_dofs(w)[i];
      
      //local_dof_indices.push_back(xdata->global_enriched_dofs(w)[i]);
    }
    if(xdata->q_points(w).size() > 0)
    {
      n_wells_inside++;
      local_dof_indices.push_back(xdata->get_well_dof_index(w)); //one more for well testing funtion
    }
  }

  //DBGMSG("number of dofs on the cell(%d): %d\n", cell->index(), n_dofs);
  
//   DBGMSG("nodes weights: ");
//   for(unsigned int w = 0; w < n_wells; w++)
//     for(unsigned int i = 0; i < dofs_per_cell; i++)
//     {
//       std::cout << node_weights[w*dofs_per_cell + i] << "  ";
//     }
//   std::cout << std::endl;

  
  #ifdef DECOMPOSED_CELL_MATRIX
    FullMatrix<double> cell_matrix_lap(n_dofs+n_wells_inside,n_dofs+n_wells_inside);
    FullMatrix<double> cell_matrix_com(n_dof+n_wells_insides,n_dofs+n_wells_inside);
  #endif
    
  cell_matrix = FullMatrix<double>(n_dofs+n_wells_inside,n_dofs+n_wells_inside);
  cell_matrix = 0;
  cell_rhs = Vector<double>(n_dofs+n_wells_inside);
  cell_rhs = 0;
  
  //temporary for shape values and gradients
  Tensor<1,2> xshape_grad;
  double xshape = 0,
         jacobian = 0, 
         jxw = 0;
  
  //vector of quadrature points on the unit square
  std::vector<Point<2> > q_points;
  //vector of quadrature points mapped to unit cell
  std::vector<Point<2> > q_points_mapped;
  
  //temporary vectors for both shape and xshape values and gradients
  std::vector<Tensor<1,2> > shape_grad_vec(n_dofs);
  std::vector<double > shape_val_vec(n_dofs+n_wells_inside,0);
  
  for(unsigned int s=0; s < squares.size(); s++)
  {
    //we are integrating on squares of the unit cell mapped from real cell!!
    q_points = squares[s].gauss->get_points();  //unit square quadrature points
    q_points_mapped = squares[s].gauss->get_points(); 
    squares[s].mapping.map_unit_to_real(q_points_mapped);  //mapped from unit square to unit cell

    Quadrature<2> temp_quad(q_points_mapped);
    FEValues<2> temp_fe_values(*fe,temp_quad, update_values | update_gradients | update_jacobians);
    temp_fe_values.reinit(cell);

    
//     //testing print of mapped q_points
//     if (s == 1)
//     {
//       DBGMSG("mapping q_points:\n");
//       for(unsigned int q=0; q < q_points.size(); q++)
//       {
//         std::cout << q_points[q] << " | ";
//       }
//       std::cout << "\n";
//       for(unsigned int q=0; q < q_points.size(); q++)
//       {
//         std::cout << q_points_mapped[q] << " | ";
//       }
//       std::cout << "\n";
//     }
//     //
//     
//     //in refinement=3 this cell is enriched but does not cross any well
//     if (cell->index() == 33)
//     {
//       DBGMSG("integration:s_jakobian: %f cell_jakobian: %f\n",squares[s].mapping.jakobian(),cell_jakobian);
//       DBGMSG("number of wells affecting this cell: %d\n", n_wells);
//     }
    //
    jacobian = squares[s].mapping.jakobian(); //square.mapping.jakobian = area of the square
    
    for(unsigned int q=0; q < q_points.size(); q++)
    {
      jxw = jacobian * temp_fe_values.jacobian(q).determinant() * squares[s].gauss->get_weights()[q];
          
      // filling FE shape values and shape gradients at first
      for(unsigned int i = 0; i < dofs_per_cell; i++)
      {
        
        shape_grad_vec[i] = temp_fe_values.shape_grad(i,q);
        
#ifdef SOURCES //----------------------------------------------------------------------------sources
        if(n_wells_inside > 0)
          shape_val_vec[i] = temp_fe_values.shape_value(i,q);
#endif
      }

      // filling xshape values and xshape gradients next
      
      unsigned int index = dofs_per_cell; //index in the vector of values and gradients
      for(unsigned int w = 0; w < n_wells; w++) //W
      { 
        //if(xdata->get_well(w)->points_inside(mapping->transform_unit_to_real_cell(cell, q_points_mapped[q])))
        //  continue;
        Well * well = xdata->get_well(w);
        //gradient of xfem function needn't be mapped (it is computed in real coordinates)
        xshape = well->global_enrich_value(mapping->transform_unit_to_real_cell(cell, q_points_mapped[q]));
        xshape_grad = well->global_enrich_grad(mapping->transform_unit_to_real_cell(cell, q_points_mapped[q]));
        
        for(unsigned int k = 0; k < dofs_per_cell; k++) //M_w
        { 
          
#ifdef SOURCES //----------------------------------------------------------------------------sources
            if(n_wells_inside > 0)
              shape_val_vec[index] = 0;   // giving zero for sure (initialized with zeros)
#endif
            shape_grad_vec[index] = 0;  // giving zero for sure (Tensor<dim> is also initialized with zeros)
            for(unsigned int l = 0; l < dofs_per_cell; l++) //M_w
            {
              
#ifdef SOURCES //----------------------------------------------------------------------------sources
              if(n_wells_inside > 0)
              {
                shape_val_vec[index] += 
                       xdata->weights(w)[l] *
                       temp_fe_values.shape_value(l,q) *             // from weight function 
                       temp_fe_values.shape_value(k,q) *
                       xshape;
              }
#endif
                       
              //gradients of shape functions need to be mapped (computed on the unit cell)
              //scale_to_unit means inverse scaling
              shape_grad_vec[index] += 
                       xdata->weights(w)[l] *
                       ( temp_fe_values.shape_grad(l,q) *            // from weight function
                         temp_fe_values.shape_value(k,q) *
                         xshape
                         +
                         temp_fe_values.shape_value(l,q) *           // from weight function
                         temp_fe_values.shape_grad(k,q) *
                         xshape
                         +
                         temp_fe_values.shape_value(l,q) *           // from weight function 
                         temp_fe_values.shape_value(k,q) *
                         xshape_grad
                       );
            } //for l
            index ++;
          //} //if
        } //for k
        
#ifdef SOURCES //----------------------------------------------------------------------------sources
        //DBGMSG("index=%d\n",index);
        //DBGMSG("shape_val_vec.size=%d\n",shape_val_vec.size());
        if(n_wells_inside > 0)
          shape_val_vec[index] = -1.0;  //testing function of the well
#endif
      } //for w
      
//       
//       unsigned int index = dofs_per_cell; //index in the vector of values and gradients
//       for(unsigned int w = 0; w < n_wells; w++) //W
//       { 
//         //if(xdata->get_well(w)->points_inside(mapping->transform_unit_to_real_cell(cell, q_points_mapped[q])))
//         //  continue;
//         Well * well = xdata->get_well(w);
//         //gradient of xfem function needn't be mapped (it is computed in real coordinates)
//         xshape = well->global_enrich_value(mapping->transform_unit_to_real_cell(cell, q_points_mapped[q]));
//         xshape_grad = well->global_enrich_grad(mapping->transform_unit_to_real_cell(cell, q_points_mapped[q]));
//         
//         for(unsigned int k = 0; k < dofs_per_cell; k++)
//         { 
//           
//         }
//         for(unsigned int k = 0; k < dofs_per_cell; k++) //Ne
//         { 
//           //if(xdata->global_enriched_dofs(w)[k] != 0) 
//           //{
// #ifdef SOURCES //----------------------------------------------------------------------------sources
//             if(n_wells_inside > 0)
//               shape_val_vec[index] = 0;   // giving zero for sure (initialized with zeros)
// #endif
//             shape_grad_vec[index] = 0;  // giving zero for sure (Tensor<dim> is also initialized with zeros)
//             for(unsigned int l = 0; l < dofs_per_cell; l++) //N
//             {
//               
// #ifdef SOURCES //----------------------------------------------------------------------------sources
//               if(n_wells_inside > 0)
//               {
//                 shape_val_vec[index] += 
//                        node_weights[w*dofs_per_cell + l] *
//                        //fe->shape_value(l, q_points_mapped[q]) *    // from weight function                                     
//                        //fe->shape_value(k, q_points_mapped[q]) *
//                        temp_fe_values.shape_value(l,q) *
//                        temp_fe_values.shape_value(k,q) *
//                        xshape;
//               }
// #endif
//                        
//               //gradients of shape functions need to be mapped (computed on the unit cell)
//               //scale_to_unit means inverse scaling
//               shape_grad_vec[index] += 
//                        node_weights[w*dofs_per_cell + l] *
//                        ( temp_fe_values.shape_grad(l,q) *
//                         //  cell_mapping.scale_inverse(
//                         //    fe->shape_grad(l, q_points_mapped[q]) ) *    // from weight function                                     
//                          //fe->shape_value(k, q_points_mapped[q]) *
//                          temp_fe_values.shape_value(k,q) *
//                          xshape
//                          +
//                          //fe->shape_value(l, q_points_mapped[q]) *   // from weight function 
//                          temp_fe_values.shape_value(l,q) *
//                          temp_fe_values.shape_grad(k,q) *
//                          //cell_mapping.scale_inverse(
//                          //   fe->shape_grad(k, q_points_mapped[q]) ) *
//                          xshape
//                          +
//                          //fe->shape_value(l, q_points_mapped[q]) *   // from weight function 
//                          //fe->shape_value(k, q_points_mapped[q]) *
//                          temp_fe_values.shape_value(l,q) *
//                          temp_fe_values.shape_value(k,q) *
//                          xshape_grad
//                        );
//             } //for l
//             index ++;
//           //} //if
//         } //for k
//         
// #ifdef SOURCES //----------------------------------------------------------------------------sources
//         //DBGMSG("index=%d\n",index);
//         //DBGMSG("shape_val_vec.size=%d\n",shape_val_vec.size());
//         if(n_wells_inside > 0)
//           shape_val_vec[index] = -1.0;  //testing function of the well
// #endif
//       } //for w          
//       
//       
      //filling cell matrix now
      //additions to matrix A,R,S from LAPLACE---------------------------------------------- LAPLACE
      
      for(unsigned int i = 0; i < n_dofs; i++)
        for(unsigned int j = 0; j < n_dofs; j++)
        {
          cell_matrix(i,j) += transmisivity * 
                              shape_grad_vec[i] *
                              shape_grad_vec[j] *
                              jxw;
                              
#ifdef DECOMPOSED_CELL_MATRIX
            cell_matrix_lap(i,j) = cell_matrix(i,j);
#endif
        }

      //addition from SOURCES--------------------------------------------------------------- SOURCES
#ifdef SOURCES
      for(unsigned int w = 0; w < n_wells; w++) //W
      {
        //this condition tests if the quadrature point lies within the well (testing function omega)
        if(xdata->get_well(w)->points_inside(mapping->transform_unit_to_real_cell(cell, q_points_mapped[q])))
        { 
          for(unsigned int i = 0; i < n_dofs+n_wells_inside; i++)
          {
            for(unsigned int j = 0; j < n_dofs+n_wells_inside; j++)
            {  
              cell_matrix(i,j) += xdata->get_well(w)->perm2aquifer() *
                                  shape_val_vec[i] *
                                  shape_val_vec[j] *
                                  jxw;
                                 
#ifdef DECOMPOSED_CELL_MATRIX
                cell_matrix_com(i,j) += 
                                  xdata->get_well(w)->perm2aquifer() *
                                  shape_val_vec[i] *
                                  shape_val_vec[j] *
                                  jxw;
#endif
                                  
            } //for j
          } //for i
        } //if
      } //for w
#endif

    
    } //for q 
  } //for s
  
  
//   std::cout << "cell_matrix" << std::endl;
//   cell_matrix.print_formatted(std::cout);
//   std::cout << std::endl;
    
  
  //------------------------------------------------------------------------------ BOUNDARY INTEGRAL
#ifdef BC_NEWTON //------------------------------------------------------------------------bc_newton
  FullMatrix<double> well_cell_matrix;
  unsigned int n_w_dofs=0;
  
  for(unsigned int w = 0; w < n_wells; w++)
  {
    if(xdata->q_points(w).size() > 0)
    {
      //DBGMSG("well number: %d\n",w);
      Well * well = xdata->get_well(w);
      //jacobian = radius of the well; weights are the same all around
      jxw = 2 * M_PI * well->radius() / well->q_points().size();
      
      //value of enriching function is constant all around the well edge
      xshape = well->global_enrich_value(well->q_points()[0]);
      //DBGMSG("q=%d  xshape=%f \n",q,xshape);
        
      shape_val_vec.clear();
      //node_weights.clear();
      //node_weights.resize(dofs_per_cell,0);
      
      
//       unsigned int n_enriched_dofs=0;
//       for(unsigned int i = 0; i < dofs_per_cell; i++)
//       {
//         if(xdata->weights(w)[i] != 0)        
//         {
//           //node_weights[i] = 1; //enriched
//           n_enriched_dofs ++;
//         }
//       }
      
      
//       DBGMSG("Printing node_weights:  [");
//         for(unsigned int a=0; a < node_weights.size(); a++)
//         {
//           std::cout << std::setw(6) << node_weights[a] << "  ";
//         }
//         std::cout << "]" << std::endl;
      
      //n_w_dofs = dofs_per_cell+n_enriched_dofs+1;
      n_w_dofs = 2*dofs_per_cell+1;
      
      well_cell_matrix.reinit(n_w_dofs, n_w_dofs);
      
      shape_val_vec.resize(n_w_dofs,0);  //unenriched, enriched, well
      
      
      //local_dof_indices.push_back(xdata->get_well_dof_indices(w)); //one more for well testing funtion
    
      //cycle over quadrature points inside the cell
      for (unsigned int q=0; q < xdata->q_points(w).size(); ++q)
      {
        Point<2> q_point = *(xdata->q_points(w)[q]);
        //transforming the quadrature point to unit cell
        Point<2> unit_point = mapping->transform_real_to_unit_cell(cell, q_point);

        // filling shape values at first
        for(unsigned int i = 0; i < dofs_per_cell; i++)
          shape_val_vec[i] = fe->shape_value(i, unit_point);
      
        // filling xshape values next
        unsigned int index = dofs_per_cell; //index in the vector of values
        
        //computing value (weight) of the ramp function
        double weight = 0;
        for(unsigned int k = 0; k < dofs_per_cell; k++) //M_w
        { 
          weight += xdata->weights(w)[k] * shape_val_vec[k]; 
        }
        
        for(unsigned int k = 0; k < dofs_per_cell; k++) //M_w
        { 
            shape_val_vec[index] = 
                     weight *
                     shape_val_vec[k] * 
                     xshape;
                     
            //DBGMSG("shape_val_vec[%d]: %f\n",index, shape_val_vec[index]);
            index ++;
        } //for k
        
        shape_val_vec[index] = -1.0;  //testing function of the well
        
        //printing enriched nodes and dofs
//         DBGMSG("Printing shape_val_vec:  [");
//         for(unsigned int a=0; a < shape_val_vec.size(); a++)
//         {
//           std::cout << std::setw(6) << shape_val_vec[a] << "  ";
//         }
//         std::cout << "]" << std::endl;
        
        for (unsigned int i=0; i < n_w_dofs; ++i)
          for (unsigned int j=0; j < n_w_dofs; ++j)
          {
              cell_matrix(i,j) += ( well->perm2aquifer() *
                                    shape_val_vec[i] *
                                    shape_val_vec[j] *
                                    jxw );
//               // for debugging
//               well_cell_matrix(i,j) += ( well->perm2aquifer() *
//                                     shape_val_vec[i] *
//                                     shape_val_vec[j] *
//                                     jxw );
          }
      } //end of iteration over q_points
    } //if
  } // for w
#endif
    
    
#ifdef DECOMPOSED_CELL_MATRIX
    DBGMSG("printing cell matrix - LAPLACE part\n");
    cell_matrix_lap.print_formatted(std::cout);
    DBGMSG("printing cell matrix - COMUNICATION part\n");
    cell_matrix_com.print_formatted(std::cout);
#endif
  
    
//     std::cout << "cell_matrix" << std::endl;
//     cell_matrix.print_formatted(std::cout);
//     std::cout << std::endl;
//     //std::cout << "well_cell_matrix" << std::endl;
//     //well_cell_matrix.print_formatted(std::cout);
//     //std::cout << std::endl;
//     
//     cell_rhs.print(std::cout);
//     std::cout << "--------------------- " << std::endl;
    
}

//*/

//OBSOLETE
/*
////////////////////////////////////////////////////// INTEGRATE_XFEM_SHIFT2 ////////////////////////////////////////////
void Adaptive_integration::integrate_xfem_shift2( FullMatrix<double> &cell_matrix, 
                                      Vector<double> &cell_rhs,
                                      std::vector<unsigned int> &local_dof_indices,
                                      const double &transmisivity)
{  
  unsigned int n_wells_inside = 0,                      // number of wells with q_points inside the cell, zero initialized
               n_wells = xdata->n_wells(),              // number of wells affecting the cell
               dofs_per_cell = fe->dofs_per_cell,
               n_vertices = GeometryInfo<2>::vertices_per_cell,
               n_dofs = n_wells*n_vertices + dofs_per_cell;     // n_wells * XFEM_n_dofs_ + FEM_n_dofs
               
  gather_w_points();
  Quadrature<2> quad(q_points_all, jxw_all);
  XFEValues<Enrichment_method::xfem_shift> xfevalues(*fe,quad, update_values 
                                                               | update_gradients 
                                                               | update_quadrature_points 
                                                               //| update_covariant_transformation 
                                                               //| update_transformation_values 
                                                               //| update_transformation_gradients
                                                               //| update_boundary_forms 
                                                               //| update_cell_normal_vectors 
                                                               | update_JxW_values 
                                                               //| update_normal_vectors
                                                               //| update_contravariant_transformation
                                                               //| update_q_points
                                                               //| update_support_points
                                                               //| update_support_jacobians 
                                                               //| update_support_inverse_jacobians
                                                               //| update_second_derivatives
                                                               //| update_hessians
                                                               //| update_volume_elements
                                                               //| update_jacobians
                                                               //| update_jacobian_grads
                                                               //| update_inverse_jacobians
                                                    );
  xfevalues.reinit(xdata);
         
  //getting unenriched local dofs indices : [FEM(dofs_per_cell), XFEM(n_wells*dofs_per_cell), WELL(n_wells)]
  local_dof_indices.clear();
  local_dof_indices.resize(dofs_per_cell);
  cell->get_dof_indices(local_dof_indices);
  
  local_dof_indices.resize(n_dofs);
  //getting enriched dof indices and well indices
  for(unsigned int w = 0; w < n_wells; w++)
  {   
    for(unsigned int i = 0; i < n_vertices; i++)
    {
      local_dof_indices[dofs_per_cell+w*n_vertices+i] = xdata->global_enriched_dofs(w)[i];
    }
    if(xdata->q_points(w).size() > 0)
    {
      n_wells_inside++;
      local_dof_indices.push_back(xdata->get_well_dof_index(w)); //one more for well testing funtion
    }
  }
    
  cell_matrix = FullMatrix<double>(n_dofs+n_wells_inside,n_dofs+n_wells_inside);
  cell_matrix = 0;
  cell_rhs = Vector<double>(n_dofs+n_wells_inside);
  cell_rhs = 0;
  
  
  //temporary vectors for both shape and xshape values and gradients
  std::vector<Tensor<1,2> > shape_grad_vec(n_dofs);
  std::vector<double > shape_val_vec(n_dofs+n_wells_inside,0);
  
  for(unsigned int q=0; q<q_points_all.size(); q++)
  { 
    // filling FE shape values and shape gradients at first
    for(unsigned int i = 0; i < dofs_per_cell; i++)
    {   
      shape_grad_vec[i] = xfevalues.shape_grad(i,q);
        
#ifdef SOURCES //----------------------------------------------------------------------------sources
        if(n_wells_inside > 0)
          shape_val_vec[i] = xfevalues.shape_value(i,q);
#endif
    }

    // filling xshape values and xshape gradients next
    unsigned int index = dofs_per_cell; //index in the vector of values and gradients
    for(unsigned int w = 0; w < n_wells; w++) //W
    {
      for(unsigned int k = 0; k < n_vertices; k++) //M_w
      { 
          
#ifdef SOURCES //----------------------------------------------------------------------------sources
        if(n_wells_inside > 0)
          shape_val_vec[index] = 0;   // giving zero for sure (initialized with zeros)
#endif
        //shape_grad_vec[index] = xfevalues.enrichment_grad(k,w,mapping->transform_unit_to_real_cell(cell, q_points_mapped[q]));
        shape_grad_vec[index] = xfevalues.enrichment_grad(k,w,q);
        index ++;
      } //for k
        
#ifdef SOURCES //----------------------------------------------------------------------------sources
      //DBGMSG("index=%d\n",index);
      //DBGMSG("shape_val_vec.size=%d\n",shape_val_vec.size());
      if(n_wells_inside > 0)
        shape_val_vec[index] = -1.0;  //testing function of the well
#endif
    } //for w
      
    //filling cell matrix now
    //additions to matrix A,R,S from LAPLACE---------------------------------------------- LAPLACE  
      for(unsigned int i = 0; i < n_dofs; i++)
        for(unsigned int j = 0; j < n_dofs; j++)
        {
          cell_matrix(i,j) += transmisivity * 
                              shape_grad_vec[i] *
                              shape_grad_vec[j] *
                              xfevalues.JxW(q); //weight of gauss * square_jacobian * cell_jacobian;
        }

      //addition from SOURCES--------------------------------------------------------------- SOURCES
#ifdef SOURCES
      for(unsigned int w = 0; w < n_wells; w++) //W
      {
        //this condition tests if the quadrature point lies within the well (testing function omega)
        if(xdata->get_well(w)->points_inside(mapping->transform_unit_to_real_cell(cell, q_points_all[q])))
        { 
          for(unsigned int i = 0; i < n_dofs+n_wells_inside; i++)
          {
            for(unsigned int j = 0; j < n_dofs+n_wells_inside; j++)
            {  
              cell_matrix(i,j) += xdata->get_well(w)->perm2aquifer() *
                                  shape_val_vec[i] *
                                  shape_val_vec[j] *
                                  xfevalues.JxW(q);
                                  
            } //for j
          } //for i
        } //if
      } //for w
#endif
  }
  
//   std::cout << "cell_matrix" << std::endl;
//   cell_matrix.print_formatted(std::cout);
//   std::cout << std::endl;
    
  
  //------------------------------------------------------------------------------ BOUNDARY INTEGRAL
#ifdef BC_NEWTON //------------------------------------------------------------------------bc_newton
  FullMatrix<double> well_cell_matrix;
  unsigned int n_w_dofs=0;
  double jxw = 0;
  
  for(unsigned int w = 0; w < n_wells; w++)
  {
    if(xdata->q_points(w).size() > 0)
    {
      std::vector<Point<2> > points(xdata->q_points(w).size());
      for (unsigned int p =0; p < points.size(); p++)
      {
        
        points[p] = mapping->transform_real_to_unit_cell(cell,*(xdata->q_points(w)[p]));
      }
      Quadrature<2> quad2 (points);
      XFEValues<Enrichment_method::xfem_shift> xfevalues2(*fe,quad2, 
                                                          update_values | update_quadrature_points);
      xfevalues2.reinit(xdata);
  
      //DBGMSG("well number: %d\n",w);
      Well * well = xdata->get_well(w);
      //jacobian = radius of the well; weights are the same all around
      jxw = 2 * M_PI * well->radius() / well->q_points().size();
      
      shape_val_vec.clear();
      
      // FEM dofs, XFEM dofs, well dof
      //n_w_dofs = dofs_per_cell+n_enriched_dofs+1;
      n_w_dofs = dofs_per_cell + n_vertices + 1;
      
      well_cell_matrix.reinit(n_w_dofs, n_w_dofs);
      shape_val_vec.resize(n_w_dofs,0);  //unenriched, enriched, well
    
      //cycle over quadrature points inside the cell
      for (unsigned int q=0; q < xdata->q_points(w).size(); ++q)
      {
        // filling shape values at first
        for(unsigned int i = 0; i < dofs_per_cell; i++)
          shape_val_vec[i] = xfevalues2.shape_value(i,q);
        
        // filling enrichment shape values
        for(unsigned int k = 0; k < n_vertices; k++)
            shape_val_vec[dofs_per_cell + k] = xfevalues2.enrichment_value(k,w,q);
        
        shape_val_vec[n_w_dofs-1] = -1.0;  //testing function of the well
        
        //printing enriched nodes and dofs
//         DBGMSG("Printing shape_val_vec:  [");
//         for(unsigned int a=0; a < shape_val_vec.size(); a++)
//         {
//           std::cout << std::setw(6) << shape_val_vec[a] << "  ";
//         }
//         std::cout << "]" << std::endl;
        
        for (unsigned int i=0; i < n_w_dofs; ++i)
          for (unsigned int j=0; j < n_w_dofs; ++j)
          {
              cell_matrix(i,j) += ( well->perm2aquifer() *
                                    shape_val_vec[i] *
                                    shape_val_vec[j] *
                                    jxw );
//               // for debugging
//               well_cell_matrix(i,j) += ( well->perm2aquifer() *
//                                     shape_val_vec[i] *
//                                     shape_val_vec[j] *
//                                     jxw );
              
          }
      }
    } //if
  } // for w
#endif
    
//     std::cout << "cell_matrix" << std::endl;
//     cell_matrix.print_formatted(std::cout);
//     std::cout << std::endl;
//     //std::cout << "well_cell_matrix" << std::endl;
//     //well_cell_matrix.print_formatted(std::cout);
//     //std::cout << std::endl;
//     
//     cell_rhs.print(std::cout);
//     std::cout << "--------------------- " << std::endl;
    
}
//*/






//OBSOLETE
/*
////////////////////////////////////////////////////// INTEGRATE_XFEM_SHIFT ////////////////////////////////////////////
void Adaptive_integration::integrate_xfem_shift( FullMatrix<double> &cell_matrix, 
                                      Vector<double> &cell_rhs,
                                      std::vector<unsigned int> &local_dof_indices,
                                      const double &transmisivity)
{  
  unsigned int n_wells_inside = 0,                      // number of wells with q_points inside the cell, zero initialized
               n_wells = xdata->n_wells(),              // number of wells affecting the cell
               dofs_per_cell = fe->dofs_per_cell,
               n_vertices = GeometryInfo<2>::vertices_per_cell,
               n_dofs = n_wells*n_vertices + dofs_per_cell;     // n_wells * XFEM_n_dofs_ + FEM_n_dofs
               
  //temporary for shape values and gradients
  Tensor<1,2> xshape_grad,
              xshape_grad_shifted;
  double xshape = 0,
         xshape_shifted = 0,
         jacobian = 0, 
         jxw = 0;
  //xshape values and gradients in nodes
         
  std::vector<double> xshape_nodes(n_wells * n_vertices);
  std::vector<Tensor<1,2> > xshape_grad_nodes(n_wells * n_vertices);
         
  //getting unenriched local dofs indices : [FEM(dofs_per_cell), XFEM(n_wells*dofs_per_cell), WELL(n_wells)]
  local_dof_indices.clear();
  local_dof_indices.resize(dofs_per_cell);
  cell->get_dof_indices(local_dof_indices);
  
  local_dof_indices.resize(n_dofs);
  //getting enriched dof indices and well indices
  for(unsigned int w = 0; w < n_wells; w++)
  {   
    for(unsigned int i = 0; i < n_vertices; i++)
    {
      local_dof_indices[dofs_per_cell+w*n_vertices+i] = xdata->global_enriched_dofs(w)[i];
     
      xshape_nodes[w*n_vertices+i] =  xdata->get_well(w)->global_enrich_value(cell->vertex(i));
      xshape_grad_nodes[w*n_vertices+i] =  xdata->get_well(w)->global_enrich_grad(cell->vertex(i));
    }
    if(xdata->q_points(w).size() > 0)
    {
      n_wells_inside++;
      local_dof_indices.push_back(xdata->get_well_dof_index(w)); //one more for well testing funtion
    }
  }
    
  cell_matrix = FullMatrix<double>(n_dofs+n_wells_inside,n_dofs+n_wells_inside);
  cell_matrix = 0;
  cell_rhs = Vector<double>(n_dofs+n_wells_inside);
  cell_rhs = 0;
  
  //vector of quadrature points on the unit square
  std::vector<Point<2> > q_points;
  //vector of quadrature points mapped to unit cell
  std::vector<Point<2> > q_points_mapped;
  
  //temporary vectors for both shape and xshape values and gradients
  std::vector<Tensor<1,2> > shape_grad_vec(n_dofs);
  std::vector<double > shape_val_vec(n_dofs+n_wells_inside,0);
  
  for(unsigned int s=0; s < squares.size(); s++)
  {
    //we are integrating on squares of the unit cell mapped from real cell!!
    q_points = squares[s].gauss->get_points();  //unit square quadrature points
    q_points_mapped = squares[s].gauss->get_points(); 
    squares[s].mapping.map_unit_to_real(q_points_mapped);  //mapped from unit square to unit cell

    Quadrature<2> temp_quad(q_points_mapped);
    FEValues<2> temp_fe_values(*fe,temp_quad, update_values | update_gradients | update_jacobians);
    temp_fe_values.reinit(cell);

    
//     //testing print of mapped q_points
//     if (s == 1)
//     {
//       DBGMSG("mapping q_points:\n");
//       for(unsigned int q=0; q < q_points.size(); q++)
//       {
//         std::cout << q_points[q] << " | ";
//       }
//       std::cout << "\n";
//       for(unsigned int q=0; q < q_points.size(); q++)
//       {
//         std::cout << q_points_mapped[q] << " | ";
//       }
//       std::cout << "\n";
//     }
//     
//     
//     //in refinement=3 this cell is enriched but does not cross any well
//     if (cell->index() == 33)
//     {
//       DBGMSG("integration:s_jakobian: %f cell_jakobian: %f\n",squares[s].mapping.jakobian(),cell_jakobian);
//       DBGMSG("number of wells affecting this cell: %d\n", n_wells);
//     }
    
    jacobian = squares[s].mapping.jakobian(); //square.mapping.jakobian = area of the square
    
    for(unsigned int q=0; q < q_points.size(); q++)
    {
      jxw = jacobian * temp_fe_values.jacobian(q).determinant() * squares[s].gauss->get_weights()[q];
          
      // filling FE shape values and shape gradients at first
      for(unsigned int i = 0; i < dofs_per_cell; i++)
      {
        
        shape_grad_vec[i] = temp_fe_values.shape_grad(i,q);
        
#ifdef SOURCES //----------------------------------------------------------------------------sources
        if(n_wells_inside > 0)
          shape_val_vec[i] = temp_fe_values.shape_value(i,q);
#endif
      }

      // filling xshape values and xshape gradients next
      
      unsigned int index = dofs_per_cell; //index in the vector of values and gradients
      for(unsigned int w = 0; w < n_wells; w++) //W
      { 
        Well* well = xdata->get_well(w);
        //gradient of xfem function needn't be mapped (it is computed in real coordinates)
        xshape = well->global_enrich_value(mapping->transform_unit_to_real_cell(cell, q_points_mapped[q]));
        xshape_grad = well->global_enrich_grad(mapping->transform_unit_to_real_cell(cell, q_points_mapped[q]));
        
        for(unsigned int k = 0; k < n_vertices; k++) //M_w
        { 
          //SHIFT
          xshape_shifted = xshape - xshape_nodes[w*n_vertices + k];
          //xshape_grad_shifted = xshape_grad - xshape_grad_nodes[w*n_vertices + k];
          //xshape_grad_shifted = xshape_grad;
          
#ifdef SOURCES //----------------------------------------------------------------------------sources
          if(n_wells_inside > 0)
            shape_val_vec[index] = 0;   // giving zero for sure (initialized with zeros)
#endif
          shape_grad_vec[index] = 0;  // giving zero for sure (Tensor<dim> is also initialized with zeros)
          for(unsigned int l = 0; l < n_vertices; l++) //M_w
          {
            
#ifdef SOURCES //----------------------------------------------------------------------------sources
            if(n_wells_inside > 0)
            {
              shape_val_vec[index] += 
                       xdata->weights(w)[l] *
                       temp_fe_values.shape_value(l,q) *             // from weight function 
                       temp_fe_values.shape_value(k,q) *
                       //xshape;
                       xshape_shifted;
            }
#endif
                 
            //gradients of shape functions need to be mapped (computed on the unit cell)
            //scale_to_unit means inverse scaling
            shape_grad_vec[index] += 
                       xdata->weights(w)[l] *
                       ( temp_fe_values.shape_grad(l,q) *            // from weight function
                         temp_fe_values.shape_value(k,q) *
                         //xshape
                         xshape_shifted
                         +
                         temp_fe_values.shape_value(l,q) *           // from weight function
                         temp_fe_values.shape_grad(k,q) *
                         //xshape
                         xshape_shifted
                         +
                         temp_fe_values.shape_value(l,q) *           // from weight function 
                         temp_fe_values.shape_value(k,q) *
                         xshape_grad
                         //xshape_grad_shifted
                       );
          } //for l
//           if (cell->index() == 60)
//           {
//             DBGMSG("shape_grad_vec[%d] = %f, %f\n", index, shape_grad_vec[index].operator[](0),shape_grad_vec[index].operator[](1));
//           }
          index ++;
        } //for k
        
#ifdef SOURCES //----------------------------------------------------------------------------sources
        //DBGMSG("index=%d\n",index);
        //DBGMSG("shape_val_vec.size=%d\n",shape_val_vec.size());
        if(n_wells_inside > 0)
          shape_val_vec[index] = -1.0;  //testing function of the well
#endif
      } //for w
      
      //filling cell matrix now
      //additions to matrix A,R,S from LAPLACE---------------------------------------------- LAPLACE
      
      for(unsigned int i = 0; i < n_dofs; i++)
        for(unsigned int j = 0; j < n_dofs; j++)
        {
          cell_matrix(i,j) += transmisivity * 
                              shape_grad_vec[i] *
                              shape_grad_vec[j] *
                              jxw;
        }

      //addition from SOURCES--------------------------------------------------------------- SOURCES
#ifdef SOURCES
      for(unsigned int w = 0; w < n_wells; w++) //W
      {
        //this condition tests if the quadrature point lies within the well (testing function omega)
        if(xdata->get_well(w)->points_inside(mapping->transform_unit_to_real_cell(cell, q_points_mapped[q])))
        { 
          for(unsigned int i = 0; i < n_dofs+n_wells_inside; i++)
          {
            for(unsigned int j = 0; j < n_dofs+n_wells_inside; j++)
            {  
              cell_matrix(i,j) += xdata->get_well(w)->perm2aquifer() *
                                  shape_val_vec[i] *
                                  shape_val_vec[j] *
                                  jxw;
                                  
            } //for j
          } //for i
        } //if
      } //for w
#endif

    
    } //for q 
  } //for s
  
  
//   std::cout << "cell_matrix" << std::endl;
//   cell_matrix.print_formatted(std::cout);
//   std::cout << std::endl;
    
  
  //------------------------------------------------------------------------------ BOUNDARY INTEGRAL
#ifdef BC_NEWTON //------------------------------------------------------------------------bc_newton
  FullMatrix<double> well_cell_matrix;
  unsigned int n_w_dofs=0;
  
  for(unsigned int w = 0; w < n_wells; w++)
  {
    if(xdata->q_points(w).size() > 0)
    {
      //DBGMSG("well number: %d\n",w);
      Well * well = xdata->get_well(w);
      //jacobian = radius of the well; weights are the same all around
      jxw = 2 * M_PI * well->radius() / well->q_points().size();
      
      //value of enriching function is constant all around the well edge
      xshape = well->global_enrich_value(well->q_points()[0]);
      //DBGMSG("q=%d  xshape=%f \n",q,xshape);
        
      shape_val_vec.clear();
      //node_weights.clear();
      //node_weights.resize(dofs_per_cell,0);
      
      
//       unsigned int n_enriched_dofs=0;
//       for(unsigned int i = 0; i < dofs_per_cell; i++)
//       {
//         if(xdata->weights(w)[i] != 0)        
//         {
//           n_enriched_dofs ++;
//         }
//       }
      
      
      // FEM dofs, XFEM dofs, well dof
      //n_w_dofs = dofs_per_cell+n_enriched_dofs+1;
      n_w_dofs = dofs_per_cell + n_vertices + 1;
      
      well_cell_matrix.reinit(n_w_dofs, n_w_dofs);
      
      shape_val_vec.resize(n_w_dofs,0);  //unenriched, enriched, well
    
      //cycle over quadrature points inside the cell
      for (unsigned int q=0; q < xdata->q_points(w).size(); ++q)
      {
        Point<2> q_point = *(xdata->q_points(w)[q]);
        //transforming the quadrature point to unit cell
        Point<2> unit_point = mapping->transform_real_to_unit_cell(cell, q_point);

        // filling shape values at first
        for(unsigned int i = 0; i < dofs_per_cell; i++)
        {
          shape_val_vec[i] = fe->shape_value(i, unit_point);
           
          //DBGMSG("shape_val_vec[%d] = %f\n", i, shape_val_vec[i]);
        }
        
        //computing value (weight) of the ramp function
        double weight = 0;
        for(unsigned int l = 0; l < n_vertices; l++)
        { 
          weight += xdata->weights(w)[l] * shape_val_vec[l]; 
        }
        
        for(unsigned int k = 0; k < n_vertices; k++)
        { 
            shape_val_vec[dofs_per_cell + k] = 
                     weight *
                     shape_val_vec[k] * 
                     ( xshape - xshape_nodes[w*n_vertices + k] );
 
            //DBGMSG("shape_val_vec[%d] = %f\n", dofs_per_cell+k, shape_val_vec[dofs_per_cell+k]);
        } //for k
        
        shape_val_vec[n_w_dofs-1] = -1.0;  //testing function of the well
        
        //printing enriched nodes and dofs
//         DBGMSG("Printing shape_val_vec:  [");
//         for(unsigned int a=0; a < shape_val_vec.size(); a++)
//         {
//           std::cout << std::setw(6) << shape_val_vec[a] << "  ";
//         }
//         std::cout << "]" << std::endl;
        
        for (unsigned int i=0; i < n_w_dofs; ++i)
          for (unsigned int j=0; j < n_w_dofs; ++j)
          {
              cell_matrix(i,j) += ( well->perm2aquifer() *
                                    shape_val_vec[i] *
                                    shape_val_vec[j] *
                                    jxw );
//               // for debugging
//               well_cell_matrix(i,j) += ( well->perm2aquifer() *
//                                     shape_val_vec[i] *
//                                     shape_val_vec[j] *
//                                     jxw );
              
          }
      } //end of iteration over q_points
    } //if
  } // for w
#endif
    
//     std::cout << "cell_matrix" << std::endl;
//     cell_matrix.print_formatted(std::cout);
//     std::cout << std::endl;
//     //std::cout << "well_cell_matrix" << std::endl;
//     //well_cell_matrix.print_formatted(std::cout);
//     //std::cout << std::endl;
//     
//     cell_rhs.print(std::cout);
//     std::cout << "--------------------- " << std::endl;
    
}
//*/





//OBSOLETE
/*
////////////////////////////////////////////////////// INTEGRATE_SGFEM ////////////////////////////////////////////
void Adaptive_integration::integrate_sgfem( FullMatrix<double> &cell_matrix, 
                                      Vector<double> &cell_rhs,
                                      std::vector<unsigned int> &local_dof_indices,
                                      const double &transmisivity)
{  
  unsigned int n_wells_inside = 0,                      // number of wells with q_points inside the cell, zero initialized
               n_wells = xdata->n_wells(),              // number of wells affecting the cell
               dofs_per_cell = fe->dofs_per_cell,
               n_vertices = GeometryInfo<2>::vertices_per_cell,
               n_dofs = dofs_per_cell;     //FEM_n_dofs (XFEM_n_dofs_ added later)
               
  QGauss<2> dummy_gauss(1);
  XFEValues<Enrichment_method::sgfem> xfevalues(*fe,dummy_gauss, update_values | update_quadrature_points);
  xfevalues.reinit(xdata);
  
  double jacobian = 0, 
         jxw = 0;
         
  //getting unenriched local dofs indices : [FEM(dofs_per_cell), XFEM(n_wells*dofs_per_cell), WELL(n_wells)]
  local_dof_indices.resize(n_dofs);
  cell->get_dof_indices(local_dof_indices);
  
  //getting enriched dof indices and well indices
  for(unsigned int w = 0; w < n_wells; w++)
  {   
    for(unsigned int i = 0; i < n_vertices; i++)
    {
      if(xdata->global_enriched_dofs(w)[i] != 0)
      {
        local_dof_indices.push_back(xdata->global_enriched_dofs(w)[i]);
        n_dofs++;
      }
    }
    if(xdata->q_points(w).size() > 0)
    {
      n_wells_inside++;
      local_dof_indices.push_back(xdata->get_well_dof_index(w)); //one more for well testing funtion
    }
  }
    
  cell_matrix = FullMatrix<double>(n_dofs+n_wells_inside,n_dofs+n_wells_inside);
  cell_matrix = 0;
  cell_rhs = Vector<double>(n_dofs+n_wells_inside);
  cell_rhs = 0;
  
  //vector of quadrature points on the unit square
  std::vector<Point<2> > q_points;
  //vector of quadrature points mapped to unit cell
  std::vector<Point<2> > q_points_mapped;
  
  //temporary vectors for both shape and xshape values and gradients
  std::vector<Tensor<1,2> > shape_grad_vec(n_dofs);
  std::vector<double > shape_val_vec(n_dofs+n_wells_inside,0);
  
  for(unsigned int s=0; s < squares.size(); s++)
  {
    //we are integrating on squares of the unit cell mapped from real cell!!
    q_points = squares[s].gauss->get_points();  //unit square quadrature points
    q_points_mapped = squares[s].gauss->get_points(); 
    squares[s].mapping.map_unit_to_real(q_points_mapped);  //mapped from unit square to unit cell

    Quadrature<2> temp_quad(q_points_mapped);
    FEValues<2> temp_fe_values(*fe,temp_quad, update_values | update_gradients | update_jacobians);
    temp_fe_values.reinit(cell);

    
//     //testing print of mapped q_points
//     if (s == 1)
//     {
//       DBGMSG("mapping q_points:\n");
//       for(unsigned int q=0; q < q_points.size(); q++)
//       {
//         std::cout << q_points[q] << " | ";
//       }
//       std::cout << "\n";
//       for(unsigned int q=0; q < q_points.size(); q++)
//       {
//         std::cout << q_points_mapped[q] << " | ";
//       }
//       std::cout << "\n";
//     }
//     
//     
//     //in refinement=3 this cell is enriched but does not cross any well
//     if (cell->index() == 33)
//     {
//       DBGMSG("integration:s_jakobian: %f cell_jakobian: %f\n",squares[s].mapping.jakobian(),cell_jakobian);
//       DBGMSG("number of wells affecting this cell: %d\n", n_wells);
//     }
    
    jacobian = squares[s].mapping.jakobian(); //square.mapping.jakobian = area of the square
    
    for(unsigned int q=0; q < q_points.size(); q++)
    {
      jxw = jacobian * temp_fe_values.jacobian(q).determinant() * squares[s].gauss->get_weights()[q];
          
      // filling FE shape values and shape gradients at first
      for(unsigned int i = 0; i < dofs_per_cell; i++)
      {
        
        shape_grad_vec[i] = temp_fe_values.shape_grad(i,q);
        
#ifdef SOURCES //----------------------------------------------------------------------------sources
        if(n_wells_inside > 0)
          shape_val_vec[i] = temp_fe_values.shape_value(i,q);
#endif
      }

      // filling xshape values and xshape gradients next
      
      unsigned int index = dofs_per_cell; //index in the vector of values and gradients
      for(unsigned int w = 0; w < n_wells; w++) //W
      { 
        for(unsigned int k = 0; k < n_vertices; k++) //M_w
        { 
          if(xdata->global_enriched_dofs(w)[k] == 0) continue;  //skip unenriched node
          
          shape_grad_vec[index] = xfevalues.enrichment_grad(k,w,mapping->transform_unit_to_real_cell(cell, q_points_mapped[q]));
            
#ifdef SOURCES //----------------------------------------------------------------------------sources
            if(n_wells_inside > 0)
            {
              shape_val_vec[index] = xfevalues.enrichment_value(k,w,mapping->transform_unit_to_real_cell(cell, q_points_mapped[q]));
            }
#endif
          index ++;
        } //for k
        
#ifdef SOURCES //----------------------------------------------------------------------------sources
        //DBGMSG("index=%d\n",index);
        //DBGMSG("shape_val_vec.size=%d\n",shape_val_vec.size());
        if(n_wells_inside > 0)
          shape_val_vec[index] = -1.0;  //testing function of the well
#endif
      } //for w
      
      //filling cell matrix now
      //additions to matrix A,R,S from LAPLACE---------------------------------------------- LAPLACE
      
      for(unsigned int i = 0; i < n_dofs; i++)
        for(unsigned int j = 0; j < n_dofs; j++)
        {
          cell_matrix(i,j) += transmisivity * 
                              shape_grad_vec[i] *
                              shape_grad_vec[j] *
                              jxw;
        }

      //addition from SOURCES--------------------------------------------------------------- SOURCES
#ifdef SOURCES
      for(unsigned int w = 0; w < n_wells; w++) //W
      {
        //this condition tests if the quadrature point lies within the well (testing function omega)
        if(xdata->get_well(w)->points_inside(mapping->transform_unit_to_real_cell(cell, q_points_mapped[q])))
        { 
          for(unsigned int i = 0; i < n_dofs+n_wells_inside; i++)
          {
            for(unsigned int j = 0; j < n_dofs+n_wells_inside; j++)
            {  
              cell_matrix(i,j) += xdata->get_well(w)->perm2aquifer() *
                                  shape_val_vec[i] *
                                  shape_val_vec[j] *
                                  jxw;
                                  
            } //for j
          } //for i
        } //if
      } //for w
#endif

    } //for q 
  } //for s
  
  
//   std::cout << "cell_matrix" << std::endl;
//   cell_matrix.print_formatted(std::cout);
//   std::cout << std::endl;
   
  
  
//------------------------------------------------------------------------------ BOUNDARY INTEGRAL
#ifdef BC_NEWTON //------------------------------------------------------------------------bc_newton
  unsigned int n_w_dofs=0;
  
  for(unsigned int w = 0; w < n_wells; w++)
  {
    if(!(xdata->q_points(w).size() > 0)) continue;
    
    //DBGMSG("well number: %d\n",w);
    Well * well = xdata->get_well(w);
    //jacobian = radius of the well; weights are the same all around
    jxw = 2 * M_PI * well->radius() / well->q_points().size();
      
        
    //how many enriched node on the cell from the well w?
    unsigned int n_enriched_dofs=0;
    for(unsigned int l = 0; l < dofs_per_cell; l++)
    {
      if(xdata->global_enriched_dofs(w)[l] != 0)
      {
        n_enriched_dofs ++;
      }
    }  
//       DBGMSG("Printing node_weights:  [");
//         for(unsigned int a=0; a < node_weights.size(); a++)
//         {
//           std::cout << std::setw(6) << node_weights[a] << "  ";
//         }
//         std::cout << "]" << std::endl;
      
    n_w_dofs = dofs_per_cell+n_enriched_dofs+1;
      
    shape_val_vec.clear();
    shape_val_vec.resize(n_w_dofs,0);  //unenriched, enriched, well
    
    //cycle over quadrature points inside the cell
    for (unsigned int q=0; q < xdata->q_points(w).size(); ++q)
    {
      Point<2> q_point = *(xdata->q_points(w)[q]);
      //transforming the quadrature point to unit cell
      Point<2> unit_point = mapping->transform_real_to_unit_cell(cell, q_point);

      // filling shape values at first
      for(unsigned int l = 0; l < dofs_per_cell; l++)
        shape_val_vec[l] = fe->shape_value(l, unit_point);
        
      // filling xshape values next
      unsigned int index = n_vertices; //index in the vector of values
      for(unsigned int k = 0; k < n_vertices; k++) 
      { 
        if(xdata->global_enriched_dofs(w)[k] != 0)
        {
          shape_val_vec[index] = xfevalues.enrichment_value(k,w,q_point);
                     
          //DBGMSG("shape_val_vec[%d]: %f\n",index, shape_val_vec[index]);
          index ++;
          }
      } //for k
        
      shape_val_vec[index] = -1.0;  //testing function of the well
        
        //printing enriched nodes and dofs
//         DBGMSG("Printing shape_val_vec:  [");
//         for(unsigned int a=0; a < shape_val_vec.size(); a++)
//         {
//           std::cout << std::setw(6) << shape_val_vec[a] << "  ";
//         }
//         std::cout << "]" << std::endl;
        
      for (unsigned int i=0; i < n_w_dofs; ++i)
        for (unsigned int j=0; j < n_w_dofs; ++j)
        {
            cell_matrix(i,j) += ( well->perm2aquifer() *
                                  shape_val_vec[i] *
                                  shape_val_vec[j] *
                                  jxw );
        }
    } //end of iteration over q_points
  } // for w
#endif

//     std::cout << "cell_matrix" << std::endl;
//     cell_matrix.print_formatted(std::cout);
//     std::cout << std::endl;
//     //std::cout << "well_cell_matrix" << std::endl;
//     //well_cell_matrix.print_formatted(std::cout);
//     //std::cout << std::endl;
//     
//     cell_rhs.print(std::cout);
//     std::cout << "--------------------- " << std::endl;  
}

//*/





//OBSOLETE
/*
////////////////////////////////////////////////////// INTEGRATE_SGFEM2 ////////////////////////////////////////////
void Adaptive_integration::integrate_sgfem2( FullMatrix<double> &cell_matrix, 
                                      Vector<double> &cell_rhs,
                                      std::vector<unsigned int> &local_dof_indices,
                                      const double &transmisivity)
{  
  unsigned int n_wells_inside = 0,                      // number of wells with q_points inside the cell, zero initialized
               n_wells = xdata->n_wells(),              // number of wells affecting the cell
               dofs_per_cell = fe->dofs_per_cell,
               n_vertices = GeometryInfo<2>::vertices_per_cell,
               n_dofs = dofs_per_cell;     //FEM_n_dofs (XFEM_n_dofs_ added later)
               
  //temporary for shape values and gradients
  Tensor<1,2> xshape_grad,
              xshape_grad_inter;
  double xshape = 0,
         xshape_inter = 0,
         jacobian = 0, 
         jxw = 0;
  //xshape values and gradients in nodes
         
  //getting unenriched local dofs indices : [FEM(dofs_per_cell), XFEM(n_wells*dofs_per_cell), WELL(n_wells)]
  local_dof_indices.clear();
  local_dof_indices.resize(dofs_per_cell);
  cell->get_dof_indices(local_dof_indices);
  
  local_dof_indices.resize(n_dofs);
  //getting enriched dof indices and well indices
  for(unsigned int w = 0; w < n_wells; w++)
  {   
    for(unsigned int i = 0; i < n_vertices; i++)
    {
      if(xdata->global_enriched_dofs(w)[i] != 0)
      {
        local_dof_indices.push_back(xdata->global_enriched_dofs(w)[i]);
        n_dofs++;
      }
    }
    if(xdata->q_points(w).size() > 0)
    {
      n_wells_inside++;
      local_dof_indices.push_back(xdata->get_well_dof_index(w)); //one more for well testing funtion
    }
  }
    
  cell_matrix = FullMatrix<double>(n_dofs+n_wells_inside,n_dofs+n_wells_inside);
  cell_matrix = 0;
  cell_rhs = Vector<double>(n_dofs+n_wells_inside);
  cell_rhs = 0;
  
  //vector of quadrature points on the unit square
  std::vector<Point<2> > q_points;
  //vector of quadrature points mapped to unit cell
  std::vector<Point<2> > q_points_mapped;
  
  //temporary vectors for both shape and xshape values and gradients
  std::vector<Tensor<1,2> > shape_grad_vec(n_dofs);
  std::vector<double > shape_val_vec(n_dofs+n_wells_inside,0);
  
  for(unsigned int s=0; s < squares.size(); s++)
  {
    //we are integrating on squares of the unit cell mapped from real cell!!
    q_points = squares[s].gauss->get_points();  //unit square quadrature points
    q_points_mapped = squares[s].gauss->get_points(); 
    squares[s].mapping.map_unit_to_real(q_points_mapped);  //mapped from unit square to unit cell

    Quadrature<2> temp_quad(q_points_mapped);
    FEValues<2> temp_fe_values(*fe,temp_quad, update_values | update_gradients | update_jacobians);
    temp_fe_values.reinit(cell);

    
//     //testing print of mapped q_points
//     if (s == 1)
//     {
//       DBGMSG("mapping q_points:\n");
//       for(unsigned int q=0; q < q_points.size(); q++)
//       {
//         std::cout << q_points[q] << " | ";
//       }
//       std::cout << "\n";
//       for(unsigned int q=0; q < q_points.size(); q++)
//       {
//         std::cout << q_points_mapped[q] << " | ";
//       }
//       std::cout << "\n";
//     }
//     
//     
//     //in refinement=3 this cell is enriched but does not cross any well
//     if (cell->index() == 33)
//     {
//       DBGMSG("integration:s_jakobian: %f cell_jakobian: %f\n",squares[s].mapping.jakobian(),cell_jakobian);
//       DBGMSG("number of wells affecting this cell: %d\n", n_wells);
//     }
    
    jacobian = squares[s].mapping.jakobian(); //square.mapping.jakobian = area of the square
    
    for(unsigned int q=0; q < q_points.size(); q++)
    {
      jxw = jacobian * temp_fe_values.jacobian(q).determinant() * squares[s].gauss->get_weights()[q];
          
      // filling FE shape values and shape gradients at first
      for(unsigned int i = 0; i < dofs_per_cell; i++)
      {
        
        shape_grad_vec[i] = temp_fe_values.shape_grad(i,q);
        
#ifdef SOURCES //----------------------------------------------------------------------------sources
        if(n_wells_inside > 0)
          shape_val_vec[i] = temp_fe_values.shape_value(i,q);
#endif
      }

      // filling xshape values and xshape gradients next
      
      unsigned int index = dofs_per_cell; //index in the vector of values and gradients
      for(unsigned int w = 0; w < n_wells; w++) //W
      { 
        Well* well = xdata->get_well(w);
        //gradient of xfem function needn't to be mapped (it is computed in real coordinates)
        xshape = well->global_enrich_value(mapping->transform_unit_to_real_cell(cell, q_points_mapped[q]));
        xshape_grad = well->global_enrich_grad(mapping->transform_unit_to_real_cell(cell, q_points_mapped[q]));
        
        //SGFEM interpolant
        xshape_inter = 0;
        xshape_grad_inter = 0;
        for(unsigned int l = 0; l < n_vertices; l++)
        {
          //xshape_inter += temp_fe_values.shape_value(l,q) * xdata->node_enrich_value(w)[l];
          //xshape_grad_inter += temp_fe_values.shape_grad(l,q) * xdata->node_enrich_value(w)[l];
          xshape_inter += temp_fe_values.shape_value(l,q) * xdata->node_enrich_value(w,l);
          xshape_grad_inter += temp_fe_values.shape_grad(l,q) * xdata->node_enrich_value(w,l);
        }
        
        for(unsigned int k = 0; k < n_vertices; k++) //M_w
        { 
          if(xdata->global_enriched_dofs(w)[k] == 0) continue;  //skip unenriched node
#ifdef SOURCES //----------------------------------------------------------------------------sources
          if(n_wells_inside > 0)
            shape_val_vec[index] = 0;   // giving zero for sure (initialized with zeros)
#endif
          shape_grad_vec[index] = 0;  // giving zero for sure (Tensor<dim> is also initialized with zeros)
            
#ifdef SOURCES //----------------------------------------------------------------------------sources
            if(n_wells_inside > 0)
            {
              shape_val_vec[index] += 
                       temp_fe_values.shape_value(k,q) *
                       (xshape - xshape_inter);
            }
#endif
            //gradients of shape functions need to be mapped (computed on the unit cell)
            //scale_to_unit means inverse scaling
            shape_grad_vec[index] += 
                         temp_fe_values.shape_value(k,q) * 
                         ( xshape_grad - xshape_grad_inter )
                         +
                         temp_fe_values.shape_grad(k,q) *
                         ( xshape - xshape_inter );
          index ++;
        } //for k
        
#ifdef SOURCES //----------------------------------------------------------------------------sources
        //DBGMSG("index=%d\n",index);
        //DBGMSG("shape_val_vec.size=%d\n",shape_val_vec.size());
        if(n_wells_inside > 0)
          shape_val_vec[index] = -1.0;  //testing function of the well
#endif
      } //for w
      
      //filling cell matrix now
      //additions to matrix A,R,S from LAPLACE---------------------------------------------- LAPLACE
      
      for(unsigned int i = 0; i < n_dofs; i++)
        for(unsigned int j = 0; j < n_dofs; j++)
        {
          cell_matrix(i,j) += transmisivity * 
                              shape_grad_vec[i] *
                              shape_grad_vec[j] *
                              jxw;
        }

      //addition from SOURCES--------------------------------------------------------------- SOURCES
#ifdef SOURCES
      for(unsigned int w = 0; w < n_wells; w++) //W
      {
        //this condition tests if the quadrature point lies within the well (testing function omega)
        if(xdata->get_well(w)->points_inside(mapping->transform_unit_to_real_cell(cell, q_points_mapped[q])))
        { 
          for(unsigned int i = 0; i < n_dofs+n_wells_inside; i++)
          {
            for(unsigned int j = 0; j < n_dofs+n_wells_inside; j++)
            {  
              cell_matrix(i,j) += xdata->get_well(w)->perm2aquifer() *
                                  shape_val_vec[i] *
                                  shape_val_vec[j] *
                                  jxw;
                                  
            } //for j
          } //for i
        } //if
      } //for w
#endif

    
    } //for q 
  } //for s
  
  
//   std::cout << "cell_matrix" << std::endl;
//   cell_matrix.print_formatted(std::cout);
//   std::cout << std::endl;
  
  
  
//------------------------------------------------------------------------------ BOUNDARY INTEGRAL
#ifdef BC_NEWTON //------------------------------------------------------------------------bc_newton
  FullMatrix<double> well_cell_matrix;
  unsigned int n_w_dofs=0;
  
  for(unsigned int w = 0; w < n_wells; w++)
  {
    if(!(xdata->q_points(w).size() > 0)) continue;
    
    //DBGMSG("well number: %d\n",w);
    Well * well = xdata->get_well(w);
    //jacobian = radius of the well; weights are the same all around
    jxw = 2 * M_PI * well->radius() / well->q_points().size();
      
    //value of enriching function is constant all around the well edge
    xshape = well->global_enrich_value(well->q_points()[0]);
    //DBGMSG("q=%d  xshape=%f \n",q,xshape);
        
      
    //how many enriched node on the cell from the well w?
    unsigned int n_enriched_dofs=0;
    for(unsigned int l = 0; l < dofs_per_cell; l++)
    {
      if(xdata->global_enriched_dofs(w)[l] != 0)
      {
        n_enriched_dofs ++;
      }
    }  
//       DBGMSG("Printing node_weights:  [");
//         for(unsigned int a=0; a < node_weights.size(); a++)
//         {
//           std::cout << std::setw(6) << node_weights[a] << "  ";
//         }
//         std::cout << "]" << std::endl;
      
    n_w_dofs = dofs_per_cell+n_enriched_dofs+1;
    well_cell_matrix.reinit(n_w_dofs, n_w_dofs);
      
    shape_val_vec.clear();
    shape_val_vec.resize(n_w_dofs,0);  //unenriched, enriched, well
    
    //cycle over quadrature points inside the cell
    for (unsigned int q=0; q < xdata->q_points(w).size(); ++q)
    {
      Point<2> q_point = *(xdata->q_points(w)[q]);
      //transforming the quadrature point to unit cell
      Point<2> unit_point = mapping->transform_real_to_unit_cell(cell, q_point);

      // filling shape values at first
      for(unsigned int l = 0; l < dofs_per_cell; l++)
        shape_val_vec[l] = fe->shape_value(l, unit_point);
      
      //SGFEM interpolation
      xshape_inter = 0;
      for(unsigned int l = 0; l < n_vertices; l++)
        //xshape_inter += shape_val_vec[l] * xdata->node_enrich_value(w)[l];
        xshape_inter += shape_val_vec[l] * xdata->node_enrich_value(w,l);
        
        
      // filling xshape values next
      unsigned int index = n_vertices; //index in the vector of values
      for(unsigned int k = 0; k < n_vertices; k++) 
      { 
        if(xdata->global_enriched_dofs(w)[k] != 0)
        {
          shape_val_vec[index] = 
                     shape_val_vec[k] * 
                     (xshape - xshape_inter);
                     
          //DBGMSG("shape_val_vec[%d]: %f\n",index, shape_val_vec[index]);
          index ++;
          }
      } //for k
        
      shape_val_vec[index] = -1.0;  //testing function of the well
        
        //printing enriched nodes and dofs
//         DBGMSG("Printing shape_val_vec:  [");
//         for(unsigned int a=0; a < shape_val_vec.size(); a++)
//         {
//           std::cout << std::setw(6) << shape_val_vec[a] << "  ";
//         }
//         std::cout << "]" << std::endl;
        
      for (unsigned int i=0; i < n_w_dofs; ++i)
        for (unsigned int j=0; j < n_w_dofs; ++j)
        {
            cell_matrix(i,j) += ( well->perm2aquifer() *
                                  shape_val_vec[i] *
                                  shape_val_vec[j] *
                                  jxw );
//               // for debugging
//               well_cell_matrix(i,j) += ( well->perm2aquifer() *
//                                     shape_val_vec[i] *
//                                     shape_val_vec[j] *
//                                     jxw );
              
        }
    } //end of iteration over q_points
  } // for w
#endif

//     std::cout << "cell_matrix" << std::endl;
//     cell_matrix.print_formatted(std::cout);
//     std::cout << std::endl;
//     //std::cout << "well_cell_matrix" << std::endl;
//     //well_cell_matrix.print_formatted(std::cout);
//     //std::cout << std::endl;
//     
//     cell_rhs.print(std::cout);
//     std::cout << "--------------------- " << std::endl;  
}
//*/





//OBSOLETE
/*
////////////////////////////////////////////////////// INTEGRATE_SGFEM3 ////////////////////////////////////////////
void Adaptive_integration::integrate_sgfem3( FullMatrix<double> &cell_matrix, 
                                      Vector<double> &cell_rhs,
                                      std::vector<unsigned int> &local_dof_indices,
                                      const double &transmisivity)
{  
  unsigned int n_wells_inside = 0,                      // number of wells with q_points inside the cell, zero initialized
               n_wells = xdata->n_wells(),              // number of wells affecting the cell
               dofs_per_cell = fe->dofs_per_cell,
               n_vertices = GeometryInfo<2>::vertices_per_cell,
               n_dofs = dofs_per_cell;   
               
  gather_w_points();
  Quadrature<2> quad(q_points_all, jxw_all);
  XFEValues<Enrichment_method::sgfem> xfevalues(*fe,quad, update_values 
                                                               | update_gradients 
                                                               | update_quadrature_points 
                                                               //| update_covariant_transformation 
                                                               //| update_transformation_values 
                                                               //| update_transformation_gradients
                                                               //| update_boundary_forms 
                                                               //| update_cell_normal_vectors 
                                                               | update_JxW_values 
                                                               //| update_normal_vectors
                                                               //| update_contravariant_transformation
                                                               //| update_q_points
                                                               //| update_support_points
                                                               //| update_support_jacobians 
                                                               //| update_support_inverse_jacobians
                                                               //| update_second_derivatives
                                                               //| update_hessians
                                                               //| update_volume_elements
                                                               //| update_jacobians
                                                               //| update_jacobian_grads
                                                               //| update_inverse_jacobians
                                                    );
  xfevalues.reinit(xdata);
         
  //getting unenriched local dofs indices : [FEM(dofs_per_cell), XFEM(n_wells*dofs_per_cell), WELL(n_wells)]
  local_dof_indices.clear();
  local_dof_indices.resize(dofs_per_cell);
  cell->get_dof_indices(local_dof_indices);
  
  local_dof_indices.resize(n_dofs);
  //getting enriched dof indices and well indices
  for(unsigned int w = 0; w < n_wells; w++)
  {   
    for(unsigned int i = 0; i < n_vertices; i++)
    {
      //local_dof_indices[dofs_per_cell+w*n_vertices+i] = xdata->global_enriched_dofs(w)[i];
      if(xdata->global_enriched_dofs(w)[i] != 0)
      {
        local_dof_indices.push_back(xdata->global_enriched_dofs(w)[i]);
        n_dofs++;
      }
    }
    if(xdata->q_points(w).size() > 0)
    {
      n_wells_inside++;
      local_dof_indices.push_back(xdata->get_well_dof_index(w)); //one more for well testing funtion
    }
  }
    
  cell_matrix = FullMatrix<double>(n_dofs+n_wells_inside,n_dofs+n_wells_inside);
  cell_matrix = 0;
  cell_rhs = Vector<double>(n_dofs+n_wells_inside);
  cell_rhs = 0;
  
  
  //temporary vectors for both shape and xshape values and gradients
  std::vector<Tensor<1,2> > shape_grad_vec(n_dofs);
  std::vector<double > shape_val_vec(n_dofs+n_wells_inside,0);
  
  for(unsigned int q=0; q<q_points_all.size(); q++)
  { 
    // filling FE shape values and shape gradients at first
    for(unsigned int i = 0; i < dofs_per_cell; i++)
    {   
      shape_grad_vec[i] = xfevalues.shape_grad(i,q);
        
#ifdef SOURCES //----------------------------------------------------------------------------sources
        if(n_wells_inside > 0)
          shape_val_vec[i] = xfevalues.shape_value(i,q);
#endif
    }

    // filling xshape values and xshape gradients next
    unsigned int index = dofs_per_cell; //index in the vector of values and gradients
    for(unsigned int w = 0; w < n_wells; w++) //W
    {
      for(unsigned int k = 0; k < n_vertices; k++) //M_w
      { 
        if(xdata->global_enriched_dofs(w)[k] == 0) continue;  //skip unenriched node  
        
#ifdef SOURCES //----------------------------------------------------------------------------sources
        if(n_wells_inside > 0)
          shape_val_vec[index] = 0;   // giving zero for sure (initialized with zeros)
#endif
        //shape_grad_vec[index] = xfevalues.enrichment_grad(k,w,mapping->transform_unit_to_real_cell(cell, q_points_mapped[q]));
        shape_grad_vec[index] = xfevalues.enrichment_grad(k,w,q);
        index ++;
      } //for k
        
#ifdef SOURCES //----------------------------------------------------------------------------sources
      //DBGMSG("index=%d\n",index);
      //DBGMSG("shape_val_vec.size=%d\n",shape_val_vec.size());
      if(n_wells_inside > 0)
        shape_val_vec[index] = -1.0;  //testing function of the well
#endif
    } //for w
      
    //filling cell matrix now
    //additions to matrix A,R,S from LAPLACE---------------------------------------------- LAPLACE  
      for(unsigned int i = 0; i < n_dofs; i++)
        for(unsigned int j = 0; j < n_dofs; j++)
        {
          cell_matrix(i,j) += transmisivity * 
                              shape_grad_vec[i] *
                              shape_grad_vec[j] *
                              xfevalues.JxW(q); //weight of gauss * square_jacobian * cell_jacobian;
        }

      //addition from SOURCES--------------------------------------------------------------- SOURCES
#ifdef SOURCES
      for(unsigned int w = 0; w < n_wells; w++) //W
      {
        //this condition tests if the quadrature point lies within the well (testing function omega)
        if(xdata->get_well(w)->points_inside(mapping->transform_unit_to_real_cell(cell, q_points_all[q])))
        { 
          for(unsigned int i = 0; i < n_dofs+n_wells_inside; i++)
          {
            for(unsigned int j = 0; j < n_dofs+n_wells_inside; j++)
            {  
              cell_matrix(i,j) += xdata->get_well(w)->perm2aquifer() *
                                  shape_val_vec[i] *
                                  shape_val_vec[j] *
                                  xfevalues.JxW(q);
                                  
            } //for j
          } //for i
        } //if
      } //for w
#endif
  }
  
//   std::cout << "cell_matrix" << std::endl;
//   cell_matrix.print_formatted(std::cout);
//   std::cout << std::endl;
    
  
  //------------------------------------------------------------------------------ BOUNDARY INTEGRAL
#ifdef BC_NEWTON //------------------------------------------------------------------------bc_newton
  FullMatrix<double> well_cell_matrix;
  unsigned int n_w_dofs=0;
  double jxw = 0;
  
  for(unsigned int w = 0; w < n_wells; w++)
  {
    if(xdata->q_points(w).size() > 0)
    {
      std::vector<Point<2> > points(xdata->q_points(w).size());
      for (unsigned int p =0; p < points.size(); p++)
      {
        
        points[p] = mapping->transform_real_to_unit_cell(cell,*(xdata->q_points(w)[p]));
      }
      Quadrature<2> quad2 (points);
      XFEValues<Enrichment_method::sgfem> xfevalues2(*fe,quad2, 
                                                          update_values | update_quadrature_points);
      xfevalues2.reinit(xdata);
  
      //DBGMSG("well number: %d\n",w);
      Well * well = xdata->get_well(w);
      //jacobian = radius of the well; weights are the same all around
      jxw = 2 * M_PI * well->radius() / well->q_points().size();
      
      //how many enriched node on the cell from the well w?
      unsigned int n_enriched_dofs=0;
      for(unsigned int l = 0; l < dofs_per_cell; l++)
      {
        if(xdata->global_enriched_dofs(w)[l] != 0)
        {
          n_enriched_dofs ++;
        }
      }  
    
      shape_val_vec.clear();
      
      // FEM dofs, XFEM dofs, well dof
      //n_w_dofs = dofs_per_cell+n_enriched_dofs+1;
      n_w_dofs = dofs_per_cell + n_enriched_dofs + 1;
      
      well_cell_matrix.reinit(n_w_dofs, n_w_dofs);
      shape_val_vec.resize(n_w_dofs,0);  //unenriched, enriched, well
    
      //cycle over quadrature points inside the cell
      for (unsigned int q=0; q < xdata->q_points(w).size(); ++q)
      {
        // filling shape values at first
        for(unsigned int i = 0; i < dofs_per_cell; i++)
          shape_val_vec[i] = xfevalues2.shape_value(i,q);
        
        // filling enrichment shape values
        for(unsigned int k = 0; k < n_vertices; k++)
        {
          if(xdata->global_enriched_dofs(w)[k] == 0) continue;  //skip unenriched node
          shape_val_vec[dofs_per_cell + k] = xfevalues2.enrichment_value(k,w,q);
        }
        
        shape_val_vec[n_w_dofs-1] = -1.0;  //testing function of the well
        
        //printing enriched nodes and dofs
//         DBGMSG("Printing shape_val_vec:  [");
//         for(unsigned int a=0; a < shape_val_vec.size(); a++)
//         {
//           std::cout << std::setw(6) << shape_val_vec[a] << "  ";
//         }
//         std::cout << "]" << std::endl;
        
        for (unsigned int i=0; i < n_w_dofs; ++i)
          for (unsigned int j=0; j < n_w_dofs; ++j)
          {
              cell_matrix(i,j) += ( well->perm2aquifer() *
                                    shape_val_vec[i] *
                                    shape_val_vec[j] *
                                    jxw );
//               // for debugging
//               well_cell_matrix(i,j) += ( well->perm2aquifer() *
//                                     shape_val_vec[i] *
//                                     shape_val_vec[j] *
//                                     jxw );
              
          }
      }
    } //if
  } // for w
#endif
    
//     std::cout << "cell_matrix" << std::endl;
//     cell_matrix.print_formatted(std::cout);
//     std::cout << std::endl;
//     //std::cout << "well_cell_matrix" << std::endl;
//     //well_cell_matrix.print_formatted(std::cout);
//     //std::cout << std::endl;
//     
//     cell_rhs.print(std::cout);
//     std::cout << "--------------------- " << std::endl;
    
}
//*/
