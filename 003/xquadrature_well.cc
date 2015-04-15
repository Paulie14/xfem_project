#include "xquadrature_well.hh"
#include "system.hh"
#include "gnuplot_i.hpp"
#include "well.hh"

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/mapping.h>

XQuadratureWell::XQuadratureWell()
: XQuadratureBase(),
  well_(nullptr),
  width_(0)
{}

XQuadratureWell::XQuadratureWell(Well* well, double width)
    : XQuadratureBase(),
    well_(well),
    width_(width)
{
    //first square
    squares_.push_back(Square(Point<2>(well_->radius(),
                                       0), 
                              Point<2>(well_->radius() + width_,
                                       2*M_PI)
                             )
                      );
    
    squares_[0].gauss = &(XQuadratureBase::quadratures_[3]);
    squares_[0].processed = true;
}

Point< 2 > XQuadratureWell::map_from_polar(Point< 2 > point)
{
    double x = point[0]*std::cos(point[1]),
           y = point[0]*std::sin(point[1]);
    return Point<2>(x,y) + well_->center();
}

Point< 2 > XQuadratureWell::map_into_polar(Point< 2 > point)
{
    Point<2> wc = well_->center();
    double r = wc.distance(point);
//     double r = std::sqrt(point.square());
    double phi = std::atan2(point[0]-wc[0], point[1]-wc[1]);
//     double phi = std::atan2(point[0], point[1]);
    return Point<2>(r,phi);
}

void XQuadratureWell::transform_square_to_real(Square& sq)
{
    if( ! sq.transformed_to_real_)
    {
        for(unsigned int i=0; i<4; i++)
        {
            sq.real_vertices_[i] = map_from_polar(sq.vertices_[i]); // map to real coordinates
        }
        sq.real_diameter_ = std::max(sq.real_vertices_[0].distance(sq.real_vertices_[2]),
                                    sq.real_vertices_[1].distance(sq.real_vertices_[3]));
        //DBGMSG("unit_diameter_=%f, real_diameter_=%f\n", sq.unit_diameter_, sq.real_diameter_);
        sq.transformed_to_real_ = true;
    }
}

void XQuadratureWell::gather_weights_points()
{
    if(quadrature_points.size() > 0) return; //do not do it again
    
    polar_quadrature_points_.reserve(squares_.size()*quadratures_[3].size());
    weights.reserve(squares_.size()*quadratures_[3].size());

    {
        for(unsigned int i = 0; i < squares_.size(); i++)
        {   
            Square square = squares_[i];
            // if no quadrature points are to be added
            if(square.gauss == nullptr) continue;
            if( *(square.gauss) == XQuadratureBase::quadratures_[0]) continue;
            
            // map from unit square to unit cell
            std::vector<Point<2> > temp(square.gauss->get_points());
            square.mapping.map_unit_to_real(temp);  
            
//             // jacobian is area of circle sector (our 'square' with polar coordinates)
//             double square_phi = std::abs(square.vertex(3)[1] - square.vertex(0)[1]);
//             double jacobian = square_phi/2 *
//                 (square.vertex(1)[0]*square.vertex(1)[0] - 
//                  square.vertex(0)[0]*square.vertex(0)[0]);
            // jacobian is area of circle sector (our 'square' with polar coordinates)
            double square_phi = std::abs(square.vertex(3)[1] - square.vertex(0)[1]);
            double jacobian = square_phi * (square.vertex(1)[0] - square.vertex(0)[0]);
                
            // gather vector of quadrature points and their weights
            for(unsigned int j = 0; j < temp.size(); j++)
            {
                polar_quadrature_points_.push_back(temp[j]);
                weights.push_back( square.gauss->weight(j) * jacobian);
            }
        }
    }
    polar_quadrature_points_.shrink_to_fit();
    weights.shrink_to_fit();
}


void XQuadratureWell::map_polar_quadrature_points_to_real(void)
{
    real_points_.clear();
    real_points_.resize(polar_quadrature_points_.size());
    for(unsigned int q = 0; q < polar_quadrature_points_.size(); q++)
    {
        real_points_[q] = map_from_polar(polar_quadrature_points_[q]);
        //weights[q] = weights[q]*polar_quadrature_points_[q][0];
    }
//     real_points_.shrink_to_fit();
}


void XQuadratureWell::refine(unsigned int max_level)
{
//     MASSERT(0, "Not implemented.");
    
    for(unsigned int t=0; t < max_level; t++)
    {
        if (refine_error()) continue;
        else break;
//         switch (refinement_type_)
//         {
//             case Refinement::edge:
//                 if (refine_edge()) continue;
//              
//             case Refinement::error:
//                 if (refine_error(alpha_tolerance_)) continue;
//              
//             case Refinement::polar:
//                 if (refine_polar()) continue;
//         }
//         break;  // if not continuing with refinement - break for cycle
    }
    
    gather_weights_points();
    map_polar_quadrature_points_to_real();
    
    // mapping squares to real coordinates
    for(auto &sq: squares_)
        transform_square_to_real(sq);
}




bool XQuadratureWell::refine_criterion_a(Square& square)
{
    //return false; // switch on and off the criterion
    transform_square_to_real(square);
    
    double min_distance = square.vertex(0)[0]*std::log(square.vertex(0)[0]);

    if( square.real_diameter() > square_refinement_criteria_factor_ * min_distance)
        return true;
    else return false;
}

bool XQuadratureWell::refine_error()
{
    unsigned int n_squares_to_refine = 0; 
    
    for(auto &sq: squares_)
    {
//         sq.refine_flag = true;
        if( refine_criterion_a(sq) )
        {
            sq.gauss = &(XQuadratureBase::quadratures_[3]);
            sq.refine_flag = true;
            n_squares_to_refine++;
        }
    }
    
    if (n_squares_to_refine == 0) 
        return false;
    else
    {
        apply_refinement(n_squares_to_refine);
        return true;
    }
}


void XQuadratureWell::create_subquadrature(XQuadratureWell& new_xquad, 
                                           const dealii::DoFHandler< 2  >::active_cell_iterator& cell,
                                           const Mapping<2> & mapping
                                          )
{
//     DBGMSG("quad size %d\n",quadrature_points.size());
    new_xquad.well_ = well_;
    new_xquad.width_ = width_;
    new_xquad.level_ = level_;
//     new_xquad.quadratures_ = quadratures_;
//     new_xquad.squares_ = squares_;
    
    new_xquad.weights.reserve(weights.size());
    new_xquad.polar_quadrature_points_.reserve(polar_quadrature_points_.size());
    new_xquad.quadrature_points.reserve(quadrature_points.size());
    new_xquad.real_points_.reserve(real_points_.size());
    
    for(unsigned int q=0; q < polar_quadrature_points_.size(); q++)
    {
//         DBGMSG("should we add point %d\n",q);
        if(cell->point_inside(real_points_[q]))
        {
//             DBGMSG("add point %d\n",q);
            new_xquad.polar_quadrature_points_.push_back(polar_quadrature_points_[q]);
            new_xquad.quadrature_points.push_back(mapping.transform_real_to_unit_cell(cell,real_points_[q]));
            new_xquad.real_points_.push_back(real_points_[q]);
            new_xquad.weights.push_back(weights[q]);
        }
    }
    
    new_xquad.weights.shrink_to_fit();
    new_xquad.polar_quadrature_points_.shrink_to_fit();
    new_xquad.quadrature_points.shrink_to_fit();
    new_xquad.real_points_.shrink_to_fit();
}


void XQuadratureWell::gnuplot_refinement(const string& output_dir, bool real, bool show)
{
//     MASSERT(real, "Point in unit cell does not make sence in this case.");
    
    
//       if(level_ < 1) return;
    DBGMSG("level = %d,  number of quadrature points = %d\n",level_, polar_quadrature_points_.size());
  
    std::string fgnuplot_ref = "adaptive_integration_refinement_",
                fgnuplot_qpoints = "adaptive_integration_qpoints_",
                script_file = "g_script_adapt_";
    
                fgnuplot_ref += ".dat";
                fgnuplot_qpoints += ".dat";
                script_file += ".p";
    try
        {
            Gnuplot g1("adaptive_integration");
        //g1.savetops("test_output");
        //g1.set_title("adaptive_integration\nrefinement");
        //g1.set_grid();
        
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
        std::cout << left << setw(53) <<  "Adaptive XFEM well refinement written in: " << fgnuplot_ref << endl;
        }
        else 
        { 
          std::cout << "Coud not write well refinement for gnuplot.\n";
        }
        myfile1.close();
        
        std::ofstream myfile2;
        myfile2.open (output_dir + fgnuplot_qpoints);
        if (myfile2.is_open()) 
        {
       
            for (unsigned int q = 0; q < polar_quadrature_points_.size(); q++)
            {
                if(real)
                    myfile2 << real_points_[q];
                else
                    myfile2 << polar_quadrature_points_[q];
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
        
        /* # parametricly plotted circle
         * set parametric
         * set trange [0:2*pi]
         * # Parametric functions for a circle
         * fx(t) = r*cos(t)
         * fy(t) = r*sin(t)
         * plot fx(t),fy(t)
         */
        strs << "set terminal x11\n";
        strs << "set size ratio -1\n";
        strs << "set parametric\n";
        strs << "set trange [0:2*pi]\n";
        if(real)
        {
          strs << "fxw(t) = " << well_->center()[0] 
              << " + "<< well_->radius() << "*cos(t)\n";
          strs << "fyw(t) = " << well_->center()[1] 
              << " + "<< well_->radius() << "*sin(t)\n";
        }
        else
        {
                  
          strs << "fxw(t) = " << 0 // real_well_center[0] 
              << " + "<< well_->radius() << "*cos(t)\n";
          strs << "fyw(t) = " << 0 // real_well_center[1] 
              << " + "<< well_->radius() << "*sin(t)\n";
        }
        
        strs << "plot \"" << fgnuplot_ref << "\" using 1:2 with lines,\\\n"
             << "\"" << fgnuplot_qpoints << "\" using 1:2 with points lc rgb \"light-blue\",\\\n";

        strs << "fxw(t),fyw(t),\\\n";
        
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
          std::cout << std::endl << "GNUPLOT output on well " << well_->center() << " ... Press ENTER to continue..." << std::endl;

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