
#ifndef comparing_h
#define comparing_h

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/base/function.h>

#include "well.hh"

using namespace dealii;

/** @brief Class contains helpful methods for comparing models.
 * 
 * Contains methods computing \f$L^2\f$ norm \f$\|\cdot\|_{L^2(\Omega)}\f$ of 
 * difference of two solutions on the given triangulation \f$\|u-v\|_2\f$, 
 * norm of the solution on the given triangulation \f$\|u\|_2\f$.
 * Method @p L2_norm_exact computes \f$\|h\|_2\f$ where \f$h\f$ is the exact solution.
 * Method @p L2_norm_diff computes \f$\|h-u\|_2\f$ where \f$h\f$ is the exact solution
 * and \f$u\f$ is given aproximating solution.
 */
class Comparing
{
public:
  /** Gets quadrature points.
    * Takes mesh, compute quadrature points and
    * removes those that lie inside the well.
    */
  static std::vector<Point<2> > get_all_quad_points(const std::string mesh_file);
  
  /// Returns \f$L^2\f$ norm of difference between two dealii vectors.
  /** Return -1 if sizes does not match.
   * @param v1
   * @param v2
   * @param tria
   */
  static double L2_norm_diff (const Vector<double> &v1, 
                              const Vector<double> &v2,
                              const Triangulation< 2 > &tria
                             );
 
  /// Returns \f$L^2\f$ norm of difference between two dealii vectors.
  /** Return -1 if sizes does not match.
   * @param input_vector
   * @param tria
   * @param well
   * @param area_radius
   */
  static double L2_norm_diff (const Vector<double> &input_vector, 
                              const Triangulation< 2 > &tria, 
                              Well *well,
                              const double &area_radius);

  /// Returns \f$L^2\f$ norm of difference between two dealii vectors.
  /** Return -1 if sizes does not match.
   * @param tria
   * @param well
   * @param area_radius
   */
  static double L2_norm_exact (const Triangulation< 2 > &tria, 
                         Well *well,
                         const double &area_radius);
  
  ///Returns \f$L^2\f$ norm of the vector on the given triangulation
  /**
   * @param input_vector
   * @param tria
   */
  static double L2_norm(const Vector< double >& input_vector, 
                        const dealii::Triangulation< 2 >& tria);
  
  /** @brief Class representing function of the exact solution.
   * 
   * We can compute value of analytical solution on a circle area with well source placed in the center.
   * The solution is given by
   *   \f{equation}
   *     u(r) = a \log(\frac{r}{R}), \quad \textrm{kde } a=\frac{P_w}{\log\frac{r_w}{R}}
   *   \f}
   * @p value returns value of exact solution in given point.
   */
  class Exact_solution : public Function<2>
    {
      public:
        /** @brief Constructor.
         * @param well is pointer to @p Well object 
         * @param radius is radius of the circle area
         */
        
        Exact_solution(Well *well, double radius);
        
        ///Returns value of exact soution in given point @p p.
        ///@param p is given point
        ///@param component is set to 0 cause it is a scalar function
        virtual double value (const Point<2>   &p,
                              const unsigned int  component = 0) const;    
      private:    
        ///Pointer to @p Well object.
        Well *well; 
        ///Constants used in computation @p value.
        double a,b;
    };
};


#endif //end of comparing_h