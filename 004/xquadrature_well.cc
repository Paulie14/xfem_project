#include "xquadrature_well.hh"
#include "system.hh"
#include "gnuplot_i.hpp"
#include "well.hh"

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/base/quadrature_lib.h>

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
    
    polar_quadrature_points_.clear();
    weights.clear();
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
    
//     double min_distance = square.vertex(0)[0]*std::log(square.vertex(0)[0]);
    double min_distance = square.real_vertex(0).distance(well_->center());
    for(unsigned int j=1; j < 4; j++)
    {
        double dist = well_->center().distance(square.real_vertex(j));
        min_distance = std::min(min_distance,dist);
    }

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


void XQuadratureWell::create_subquadrature(XQuadratureWell* new_xquad, 
                                           const dealii::DoFHandler< 2  >::active_cell_iterator& cell,
                                           const Mapping<2> & mapping
                                          )
{
//     DBGMSG("quad size %d\n",quadrature_points.size());
    new_xquad->well_ = well_;
    new_xquad->width_ = width_;
    new_xquad->level_ = level_;
//     new_xquad->quadratures_ = quadratures_;
//     new_xquad->squares_ = squares_;
    
    new_xquad->weights.clear();
    new_xquad->polar_quadrature_points_.clear();
    new_xquad->quadrature_points.clear();
    new_xquad->real_points_.clear();
    
    new_xquad->weights.reserve(weights.size());
    new_xquad->polar_quadrature_points_.reserve(polar_quadrature_points_.size());
    new_xquad->quadrature_points.reserve(quadrature_points.size());
    new_xquad->real_points_.reserve(real_points_.size());
    
    for(unsigned int q=0; q < polar_quadrature_points_.size(); q++)
    {
//         DBGMSG("should we add point %d\n",q);
        if(cell->point_inside(real_points_[q]))
        {
//             DBGMSG("add point %d\n",q);
            new_xquad->polar_quadrature_points_.push_back(polar_quadrature_points_[q]);
            new_xquad->quadrature_points.push_back(mapping.transform_real_to_unit_cell(cell,real_points_[q]));
            new_xquad->real_points_.push_back(real_points_[q]);
            new_xquad->weights.push_back(weights[q]);
        }
    }
    
    new_xquad->weights.shrink_to_fit();
    new_xquad->polar_quadrature_points_.shrink_to_fit();
    new_xquad->quadrature_points.shrink_to_fit();
    new_xquad->real_points_.shrink_to_fit();
}


void XQuadratureWell::gnuplot_refinement(const string& output_dir, bool real, bool show)
{
//     MASSERT(real, "Point in unit cell does not make sence in this case.");
    
#if VERBOSE_QUAD
    DBGMSG("level = %d,  number of quadrature points = %d\n",level_, polar_quadrature_points_.size());
#endif  
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
#if VERBOSE_QUAD        
        std::cout << left << setw(53) <<  "Adaptive XFEM well refinement written in: " << fgnuplot_ref << endl;
#endif        
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
#if VERBOSE_QUAD        
            std::cout << left << setw(53) <<  "Quadrature points written in: " << fgnuplot_qpoints << std::endl;
#endif            
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
          strs << "sx = " << well_->center()[0]  << "\n";
          strs << "sy = " << well_->center()[1]  << "\n";
        }
        else
        {
          strs << "sx = " << 0  << "\n";
          strs << "sy = " << 0  << "\n";
        }
        strs << "r = " << well_->radius() << "\n";
        strs << "w = " << width_ << "\n";
        strs << "fxw(t) = sx + r*cos(t)\n";
        strs << "fyw(t) = sy + r*sin(t)\n";
        strs << "gxw(t) = sx + (r+w)*cos(t)\n";
        strs << "gyw(t) = sy + (r+w)*sin(t)\n";
        
        strs << "set style line 1 lt 2 lw 2 lc rgb 'blue'\n"
             << "set style line 2 lt 1 lw 2 lc rgb '#66A61E'\n";
        strs << "plot '" << fgnuplot_qpoints << "' using 1:2 with points lc rgb 'light-blue' title 'quadrature points' ,\\\n"
             << "'" << fgnuplot_ref << "' using 1:2 with lines lc rgb 'red' title 'refinement' ,\\\n"
             << "fxw(t),fyw(t) ls 1 title 'well edge',\\\n"
             << "gxw(t),gyw(t) ls 2 title 'well band'\n";

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
#if VERBOSE_QUAD          
          std::cout << left << setw(53) << "Gnuplot script for adaptive refinement written in: " << script_file << endl;
#endif          
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





XQuadratureWellLog::XQuadratureWellLog()
: XQuadratureWell(nullptr, 0),
    n_phi_(100),
    gauss_degree_(5)
{}

XQuadratureWellLog::XQuadratureWellLog(Well* well, double width, unsigned int n_phi, unsigned int gauss_degree)
    : XQuadratureWell(well, width),
    n_phi_(n_phi),
    gauss_degree_(gauss_degree)
{}

void XQuadratureWellLog::transform_square_to_real(Square& square)
{
    MASSERT(0, "This cannot be called for XQuadratureWellLog.");
}

bool XQuadratureWellLog::refine_error()
{
    double phi_step = 2*M_PI/n_phi_;
    unsigned int quad_order = gauss_degree_;
    
    // create quadrature on [0,1]
    QGauss<1> gauss(quad_order);
//     QGaussLogR<1> gauss(quad_order, Point<1>(0.0), width_, true);
    
    // map points from [0,1] to [rho, rho + bandwidth]
    std::vector<double> q_points_mapped(quad_order);
    for(unsigned int j=0; j < quad_order; j++)
        {
//             DBGMSG("gauss point weight: %e \t%e\n",gauss_o.point(j)[0], gauss_o.weight(j));
            q_points_mapped[j] = gauss.point(j)[0] * width_ + well_->radius();
        }
    
    for(unsigned int i=0; i < n_phi_; i++)
    {
        // offset due to create subquadrature on cells - on perpendicular cells it avoids many quad points to
        // be on the edge and therefore included in more cells
        double phi = numeric_limits< double >::epsilon() + i * phi_step;    
        for(unsigned int j=0; j < quad_order; j++)
        {
            polar_quadrature_points_.push_back(Point<2>(q_points_mapped[j],phi));
            weights.push_back(gauss.weight(j)*phi_step*width_);
        }
    }
    
    return false;
}

void XQuadratureWellLog::gather_weights_points()
{
    // do nothing
}

void XQuadratureWellLog::refine(unsigned int max_level)
{
    refine_error();
    map_polar_quadrature_points_to_real();
}


void XQuadratureWellLog::gnuplot_refinement(const string& output_dir, bool real, bool show)
{
//     MASSERT(real, "Point in unit cell does not make sence in this case.");
    
#if VERBOSE_QUAD
    DBGMSG("level = %d,  number of quadrature points = %d\n",level_, polar_quadrature_points_.size());
#endif  
    std::string fgnuplot_qpoints = "polar_integration_qpoints_",
                script_file = "g_script_polar_";
    
                fgnuplot_qpoints += ".dat";
                script_file += ".p";
    try
    {
        Gnuplot g1("polar_integration");
        //g1.savetops("test_output");
        //g1.set_title("adaptive_integration\nrefinement");
        //g1.set_grid();        
        
        std::ofstream myfile2;
        myfile2.open (output_dir + fgnuplot_qpoints);
        if (myfile2.is_open()) 
        {
            if(real)
            {
                for (unsigned int q = 0; q < real_points_.size(); q++)
                    myfile2 << real_points_[q] << "\n";
            }
            else
            {
                for (unsigned int q = 0; q < polar_quadrature_points_.size(); q++)
                    myfile2 << polar_quadrature_points_[q]  << "\n";
            } 
            
            
#if VERBOSE_QUAD        
            std::cout << left << setw(53) <<  "Quadrature points written in: " << fgnuplot_qpoints << std::endl;
#endif            
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
            strs << "sx = " << well_->center()[0]  << "\n";
            strs << "sy = " << well_->center()[1]  << "\n";
            strs << "r = " << well_->radius() << "\n";
            strs << "w = " << width_ << "\n";
            strs << "fxw(t) = sx + r*cos(t)\n";
            strs << "fyw(t) = sy + r*sin(t)\n";
            strs << "gxw(t) = sx + (r+w)*cos(t)\n";
            strs << "gyw(t) = sy + (r+w)*sin(t)\n";
            strs << "set style line 1 lt 2 lw 2 lc rgb 'blue'\n"
             << "set style line 2 lt 1 lw 2 lc rgb '#66A61E'\n";
            
            strs << "plot '" << fgnuplot_qpoints << "' using 1:2 with points lc rgb 'light-blue' title 'quadrature points' ,\\\n"
             << "fxw(t),fyw(t) ls 1 title 'well edge',\\\n"
             << "gxw(t),gyw(t) ls 2 title 'well band'\n";
        }
        else
        {
            // do not print wells in polar coordinates
        }
        //saving gnuplot script
        std::ofstream myfile3;
        myfile3.open (output_dir + script_file);
        if (myfile3.is_open()) 
        {
          // header
          myfile3 << "# Gnuplot script for printing polar quadrature around a well.\n" <<
                     "# Made by Pavel Exner.\n#\n" <<
                     "# Run the script in gnuplot:\n" <<
                     "# > load \"" << script_file << "\"\n#\n" <<
                     "# Data files used:\n" << 
                     "# " << fgnuplot_qpoints << "\n" 
                     "#\n#" << std::endl;
          // script
          myfile3 << strs.str() << std::endl;
#if VERBOSE_QUAD          
          std::cout << left << setw(53) << "Gnuplot script for adaptive refinement written in: " << script_file << endl;
#endif          
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










XQuadratureWellBand::XQuadratureWellBand()
: XQuadratureWell(nullptr, 0)
{}

XQuadratureWellBand::XQuadratureWellBand(Well* well, double width, unsigned int gauss_degree)
    : XQuadratureWell(well, width),
    gauss_(new QGauss<2>(gauss_degree))
{}

bool XQuadratureWellBand::refine_error()
{
    //compute number of squares: circumference of band / width of band
    unsigned int n_squares = std::floor(2*M_PI*(well_->radius() + width_) / width_);
    
    double phi_step = 2*M_PI/n_squares;
    
    std::cout << "n_squares to create: " << n_squares << std::endl;
    squares_.clear();
    for(unsigned int s=0; s < n_squares; s++)
    {
        double phi = numeric_limits< double >::epsilon() + s * phi_step;
        
        //first square
        squares_.push_back(Square(Point<2>(well_->radius(),
                                        phi), 
                                Point<2>(well_->radius() + width_,
                                        phi + phi_step)
                                )
                        );
    
        squares_.back().gauss = gauss_;
        squares_.back().processed = true;
    }
    return false;
}

void XQuadratureWellBand::gnuplot_refinement(const string& output_dir, bool real, bool show)
{   
#if VERBOSE_QUAD
    DBGMSG("level = %d,  number of quadrature points = %d\n",level_, polar_quadrature_points_.size());
#endif  
    std::string fgnuplot_ref = "adaptive_integration_refinement_",
                fgnuplot_qpoints = "adaptive_integration_qpoints_",
                script_file = "g_script_adapt_";
    
                fgnuplot_ref += ".dat";
                fgnuplot_qpoints += ".dat";
                script_file += ".p";
    try {
        Gnuplot g1("adaptive_integration");
        
        std::ofstream myfile1;
        myfile1.open (output_dir + fgnuplot_ref);
        if (myfile1.is_open()) 
        {
       
        for (unsigned int i = 0; i < squares_.size(); i++)
        {
            
            if(real)
            {
                myfile1 << squares_[i].real_vertex(0) << "\n"
                        << squares_[i].real_vertex(1) << "\n\n";
            }
            else
            {
                myfile1 << squares_[i].vertex(0) << "\n"
                        << squares_[i].vertex(1) << "\n\n";
            }
        }
#if VERBOSE_QUAD        
        std::cout << left << setw(53) <<  "Adaptive XFEM well refinement written in: " << fgnuplot_ref << endl;
#endif        
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
#if VERBOSE_QUAD        
            std::cout << left << setw(53) <<  "Quadrature points written in: " << fgnuplot_qpoints << std::endl;
#endif            
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
        strs << "#set terminal postscript eps enhanced color font 'Helvetica,15' linewidth 2\n";
        strs << "#set output 'polar_band_refinement.eps'\n";
        strs << "set size ratio -1\n";
        strs << "set parametric\n";
        strs << "set trange [0:2*pi]\n";
        if(real)
        {
          strs << "sx = " << well_->center()[0]  << "\n";
          strs << "sy = " << well_->center()[1]  << "\n";
        }
        else
        {
          strs << "sx = " << 0  << "\n";
          strs << "sy = " << 0  << "\n";
        }
        strs << "r = " << well_->radius() << "\n";
        strs << "w = " << width_ << "\n";
        strs << "fxw(t) = sx + r*cos(t)\n";
        strs << "fyw(t) = sy + r*sin(t)\n";
        strs << "gxw(t) = sx + (r+w)*cos(t)\n";
        strs << "gyw(t) = sy + (r+w)*sin(t)\n";
        
        strs << "set style line 1 lt 2 lw 2 lc rgb 'blue'\n"
             << "set style line 2 lt 1 lw 2 lc rgb '#66A61E'\n";
        strs << "plot '" << fgnuplot_qpoints << "' using 1:2 with points lc rgb 'light-blue' title 'quadrature points' ,\\\n"
             << "'" << fgnuplot_ref << "' using 1:2 with lines lc rgb 'red' title 'refinement' ,\\\n"
             << "fxw(t),fyw(t) ls 1 title 'well edge',\\\n"
             << "gxw(t),gyw(t) ls 2 title 'well band'\n";
        
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
#if VERBOSE_QUAD          
          std::cout << left << setw(53) << "Gnuplot script for adaptive refinement written in: " << script_file << endl;
#endif          
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
