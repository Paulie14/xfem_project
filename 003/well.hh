#ifndef Well_h
#define Well_h


#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>

#include "massert.h"

using namespace dealii;

/** @brief Class representing the well.
 * Contains the geometry of the well and physical parameters of the well.
 */
class Well
{
  public:
    
    /// Default constructor.
    Well ();
    
    /// Main constructor with well's geometry parameters definition.
    Well (double r, Point<2> cent);
    
    /// Additional simple constructor.
    Well (double r, Point<2> cent, double perm2fer, double perm2tard);
      
    ///constructor
    Well (Well *well);    

    /// @name Getters
    //@{
    
    /// returns radius
    double radius();
  
    /// returns center
    Point<2> center();
    
    /// Returns permeability to @p m-th aquifer.
    double perm2aquifer(unsigned int m);
    
    /// Returns permeability to @p m-th aquitard.
    double perm2aquitard(unsigned int m);
    
    /// returns pressure in the well
    double pressure();
    
    /// returns pressure in the well (piezometric help)
    double pressure(const Point<2> &p);
    
    /// Is true if the pressure at the top is set.
    bool is_pressure_set();
    
    const std::vector<Point<2> > &q_points();
    
    bool is_active();
    //@}
  
  
    /// @name Setters
    //@{
    /// Sets permeability to @p m-th aquifer.
    void set_perm2aquifer(unsigned int m, double perm);
    
    /// Sets permeability to @p m-th aquitard.
    void set_perm2aquitard(unsigned int m, double perm);

    
    /// Sets permeability to all aquifers.
    void set_perm2aquifer(std::vector<double> perm);
    
    /// Sets permeability to all aquitards.
    void set_perm2aquitard(std::vector<double> perm);
    
    /// sets pressure
    void set_pressure(double press);
    
    // Activates the well.
    void set_active();
    
    // Deactivates the well - has no influence in the model.
    void set_inactive();
    //@}
    
    /// Computes circumference of the well_edge
    double circumference();
    
    /// returns true if the given point lies within the well radius
    bool points_inside(const Point<2> &point);
    
    /// computes @p n points equally distributed on the well boundary
    void evaluate_q_points(const unsigned int &n);
    
    /** Returns value of the global enrichment function.
     *  @param point is point at which the global enrichment function is evaluated
     */
    double global_enrich_value(const Point<2> &point);
    
    /** Returns gradient value of the global enrichment function.
     *  @param point is point at which the gradient of the global enrichment function is evaluated
     */
    Tensor<1,2,double> global_enrich_grad(const Point<2> &point);
  
  protected:
    
    /// True, if the well should be active and included in computation.
    bool active_;
    
    /// radius of the well
    double radius_;
    
    /// pressure at the top of the well (Dirichlet boundary condition)
    double pressure_;
    
    /// Flag is true if the pressure at the top is set.
    bool pressure_set_;
    
    /// position of the center of the well
    /** There is a reason that there are no setting methods. 
     *  Models can be run in cycles - if triangulation is not changed
     *  then it is considered that no geometric parameters of the wells are changed.
     */
    Point<2> center_;
    
    /// permeability between the well and aquifer
    std::vector<double> perm2aquifer_;
    
    /// permeability between in the well between aquitards
    std::vector<double> perm2aquitard_;
    
    /// points placed around the circle by @p evaluate_q_points function
    std::vector<Point<2> > q_points_;
    
};



/****************************************            Implementation          ********************************/
inline double Well::radius()
{return radius_;}

inline Point<2> Well::center()
{return center_;}

inline double Well::perm2aquifer(unsigned int m)
{   MASSERT(m < perm2aquifer_.size(), "Aquifer index exceeded."); 
    return perm2aquifer_[m]; }

inline double Well::perm2aquitard(unsigned int m)
{   MASSERT(m < perm2aquitard_.size(), "Aquitard index exceeded."); 
    return perm2aquitard_[m]; }

inline double Well::pressure()
{   MASSERT(pressure_set_, "Pressure not set.");
    return pressure_; }

inline double Well::pressure(const Point<2> &p)
{   MASSERT(pressure_set_, "Pressure not set.");
    return pressure_ - p[1]; }

inline bool Well::is_pressure_set()
{return pressure_set_;}

inline const std::vector<Point<2> > & Well::q_points()
{return q_points_;}

inline bool Well::is_active()
{ return active_;}


inline void Well::set_perm2aquifer(unsigned int m, double perm)
{   MASSERT(m < perm2aquifer_.size(), "Aquifer index exceeded."); 
    perm2aquifer_[m] = perm;}

inline void Well::set_perm2aquitard(unsigned int m, double perm)
{   MASSERT(m < perm2aquitard_.size(), "Aquitard index exceeded."); 
    perm2aquitard_[m] = perm;}

inline void Well::set_perm2aquifer(std::vector<double> perm)
{ perm2aquifer_ = perm;}

inline void Well::set_perm2aquitard(std::vector<double> perm)
{ perm2aquitard_ = perm;}

inline void Well::set_pressure(double press)
{   pressure_ = press;
    pressure_set_ = true;
}

inline void Well::set_active()
{ active_ = true; }

inline void Well::set_inactive()
{ active_ = false; }


inline bool Well::points_inside(const Point<2> &point)
{
  return point.distance(center_) <= radius_;
}


#endif // Exact_solution_h