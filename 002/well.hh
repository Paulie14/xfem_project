#ifndef Well_h
#define Well_h

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/dofs/dof_tools.h>

using namespace dealii;

/** @brief Class representing the well.
 * Contains the geometry of the well and physical parameters of the well.
 */
class Well
{
  public:
    
    ///Default constructor.
    Well () 
      {}
    
    ///main constructor with well's parameters definition
    Well (double r, Point<2> cent, double perm2fer, double perm2tard);
      
    ///constructor
    Well (Well *well);    

    /// returns radius
    inline double radius()
    {return radius_;}
  
    /// returns center
    inline Point<2> center()
    {return center_;}
    
    /// returns permeability to aquifer
    inline double perm2aquifer()
    {return perm2aquifer_;}
    
    /// returns permeability to aquitard
    inline double perm2aquitard()
    {return perm2aquitard_;}
  
    /// returns pressure in the well
    inline double pressure()
    {return pressure_;}
    
    /// returns pressure in the well
    inline double pressure(const Point<2> &p)
    {return pressure_ - p[1];}
    
    inline const std::vector<Point<2> > &q_points()
    {return q_points_;}
  
  
  
    /// sets permeability to aquifer
    inline void set_perm2aquifer(const double &perm)
    { perm2aquifer_ = perm;}
    
    /// sets permeability to aquitard
    inline void set_perm2aquitard(const double &perm)
    { perm2aquitard_ = perm;}
    
    /// sets pressure
    inline void set_pressure(const double &press)
    { pressure_ = press;}
    
    
    
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
    
    /// position of the center of the well
    /** There is a reason that there are no setting methods. 
     *  Models can be run in cycles - if triangulation is not changed
     *  then it is considered that no geometric parameters of the wells are changed.
     */
    Point<2> center_;
    
    /// permeability between the well and aquifer
    double perm2aquifer_;
    
    /// permeability between in the well between aquitards
    double perm2aquitard_;
    
    /// pressure at the top of the well (Dirichlet boundary condition)
    double pressure_;
    
    /// points placed around the circle by @p evaluate_q_points function
    std::vector<Point<2> > q_points_;
    
};



#endif // Exact_solution_h