#ifndef Well_h
#define Well_h

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/dofs/dof_tools.h>

#include "massert.h"

using namespace dealii;

/** @brief Class representing the well.
 * Contains the geometry of the well and physical parameters of the well.
 */
class Well
{
  public:
    
    /// Default constructor.
    Well () 
      {}
    
    /// Main constructor with well's geometry parameters definition.
    Well (double r, Point<2> cent);
    
    /// Additional simple constructor.
    Well (double r, Point<2> cent, double perm2fer, double perm2tard);
      
    ///constructor
    Well (Well *well);    

    /// @name Getters
    //@{
    
    /// returns radius
    inline double radius()
    {return radius_;}
  
    /// returns center
    inline Point<2> center()
    {return center_;}
    
    /// Returns permeability to @p m-th aquifer.
    inline double perm2aquifer(unsigned int m)
    {   MASSERT(m < perm2aquifer_.size(), "Aquifer index exceeded."); 
        return perm2aquifer_[m]; }
    
    /// Returns permeability to @p m-th aquitard.
    inline double perm2aquitard(unsigned int m)
    {   MASSERT(m < perm2aquitard_.size(), "Aquitard index exceeded."); 
        return perm2aquitard_[m]; }
  
    /// returns pressure in the well
    inline double pressure()
    {   MASSERT(pressure_set_, "Pressure not set.");
        return pressure_; }
    
    /// returns pressure in the well (piezometric help)
    inline double pressure(const Point<2> &p)
    {   MASSERT(pressure_set_, "Pressure not set.");
        return pressure_ - p[1]; }
    
    /// Is true if the pressure at the top is set.
    inline bool is_pressure_set()
    {return pressure_set_;}
    
    inline const std::vector<Point<2> > &q_points()
    {return q_points_;}
    //@}
  
  
    /// @name Setters
    //@{
    /// Sets permeability to @p m-th aquifer.
    inline void set_perm2aquifer(unsigned int m, double perm)
    {   MASSERT(m < perm2aquifer_.size(), "Aquifer index exceeded."); 
        perm2aquifer_[m] = perm;}
    
    /// Sets permeability to @p m-th aquitard.
    inline void set_perm2aquitard(unsigned int m, double perm)
    {   MASSERT(m < perm2aquitard_.size(), "Aquitard index exceeded."); 
        perm2aquitard_[m] = perm;}
    
    /// Sets permeability to all aquifers.
    inline void set_perm2aquifer(std::vector<double> perm)
    { perm2aquifer_ = perm;}
    
    /// Sets permeability to all aquitards.
    inline void set_perm2aquitard(std::vector<double> perm)
    { perm2aquitard_ = perm;}
    
    /// sets pressure
    inline void set_pressure(double press)
    {   pressure_ = press;
        pressure_set_ = true;
    }
    //@}
    
    
    /// returns true if the given point lies within the well radius
    inline bool points_inside(const Point<2> &point)
    {
      return point.distance(center_) <= radius_;
    }
    
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



#endif // Exact_solution_h