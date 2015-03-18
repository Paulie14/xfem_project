
#ifndef comparing_h
#define comparing_h


#include <deal.II/base/function.h>

#include "well.hh"

//forward declaration
namespace dealii{
    template<int,int> class Triangulation;
}

/** Groups together the exact solutions and source terms.
 * All the classes are descendants of abstract templated class @p dealii::Function<2>
 */
namespace Solution
{
  /** @brief Class representing function of the exact solution.
   * 
   * We can compute value of analytical solution on a circle area with well source placed in the center.
   * The solution is given by
   *   \f{equation}
   *     u(r) = a \log(\frac{r}{R}), \quad \textrm{kde } a=\frac{P_w}{\log\frac{r_w}{R}}
   *   \f}
   * @p value returns value of exact solution in given point.
   */
  class ExactBase : public dealii::Function<2>
    {
      public:
        /** @brief Constructor.
         * @param well is pointer to @p Well object 
         * @param radius is radius of the circle area
         */
        
        ExactBase(Well *well, double radius, double p_dirichlet = 0);
        
        ///Returns value of exact soution in given point @p p.
        ///@param p is given point
        ///@param component is set to 0 cause it is a scalar function
        virtual double value (const dealii::Point<2>   &p,
                              const unsigned int  component = 0) const = 0;
                              
        virtual dealii::Tensor<1,2> grad (const dealii::Point<2>   &p,
                             const unsigned int  component = 0) const = 0;    
                              
        inline double a() {return a_;}
        inline double b() {return b_;}
        inline double radius() {return radius_;}
        inline double p_dirichlet() {return p_dirichlet_;}
        inline Well* well() {return well_;}
      protected:    
        ///Pointer to @p Well object.
        Well *well_; 
        ///Constants used in computation @p value.
        double radius_, p_dirichlet_, a_, b_;
        /// Index of aquifer. Only m_=0 is used.
        unsigned int m_;
    };

    
    class ExactSolutionZero : public ExactBase
    {
    public:
      ExactSolutionZero() : ExactBase(new Well(), 0, 0)
      {}
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
      dealii::Tensor<1,2> grad (const dealii::Point<2>   &p,
                                const unsigned int  component = 0) const override;   
    };
    
    class ExactSolution : public ExactBase
    {
    public:
      ExactSolution(Well *well, double radius, double p_dirichlet = 0) : ExactBase(well, radius, p_dirichlet) {}
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
      dealii::Tensor<1,2> grad (const dealii::Point<2>   &p,
                                const unsigned int  component = 0) const override;   
    };
    
    class ExactSolution1 : public ExactBase
    {
    public:
      ExactSolution1(Well *well, double radius, double k, double amplitude);
        
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
      dealii::Tensor<1,2> grad (const dealii::Point<2>   &p,
                                const unsigned int  component = 0) const override;   
    protected:
        double k_, amplitude_;
        
    friend class Source1;
    };
    
    class Source1 : public ExactSolution1
    {
    public:
      //Source1(Well *well, double radius, double k) : ExactSolution1(well, radius, k) {}
      Source1(ExactSolution1 &ex_sol) 
        : ExactSolution1(ex_sol.well_, ex_sol.radius_, ex_sol.k_, ex_sol.amplitude_) {}
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
    };
    
    class ExactSolution2 : public ExactBase
    {
    public:
      ExactSolution2(Well *well, double radius, double k) : ExactBase(well, radius, 0), k_(k) {}
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
      dealii::Tensor<1,2> grad (const dealii::Point<2>   &p,
                                const unsigned int  component = 0) const override;   
    protected:
        double k_;
    
    friend class Source2;
    };
    
    class Source2 : public ExactSolution2
    {
    public:
      Source2(Well *well, double radius, double k) : ExactSolution2(well, radius, k) {}
      Source2(ExactSolution2 &ex_sol) : ExactSolution2(ex_sol.well_, ex_sol.radius_, ex_sol.k_) {}
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
    };


    class ExactSolution3 : public ExactBase
    {
    public:
      ExactSolution3(Well *well, double radius, double k, double amplitude);
        
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
      dealii::Tensor<1,2> grad (const dealii::Point<2>   &p,
                                const unsigned int  component = 0) const override;   
    protected:
        double k_, amplitude_;
        
    friend class Source3;
    };
    
    class Source3 : public ExactSolution3
    {
    public:
      //Source1(Well *well, double radius, double k) : ExactSolution1(well, radius, k) {}
      Source3(ExactSolution3 &ex_sol) 
        : ExactSolution3(ex_sol.well_, ex_sol.radius_, ex_sol.k_, ex_sol.amplitude_) {}
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
    };
}  
using namespace Solution;
    
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
  /// Returns \f$L^2\f$ norm of difference between two dealii vectors.
  /** Return -1 if sizes does not match.
   * @param v1
   * @param v2
   * @param tria
   */
  static double L2_norm_diff (const dealii::Vector<double> &v1, 
                              const dealii::Vector<double> &v2,
                              const dealii::Triangulation< 2 > &tria
                             );
 
  /// Returns \f$L^2\f$ norm of difference between two dealii vectors.
  /** Return -1 if sizes does not match.
   * @param input_vector
   * @param tria
   * @param well
   * @param area_radius
   */
  static double L2_norm_diff (const dealii::Vector<double> &input_vector, 
                              const dealii::Triangulation< 2 > &tria, 
                              dealii::Function<2>* exact_solution);

  /// Returns \f$L^2\f$ norm of difference between two dealii vectors.
  /** Return -1 if sizes does not match.
   * @param tria
   * @param well
   * @param area_radius
   */
  static double L2_norm_exact (const dealii::Triangulation< 2 > &tria, 
                               dealii::Function<2>* exact_solution);
  
  ///Returns \f$L^2\f$ norm of the vector on the given triangulation
  /**
   * @param input_vector
   * @param tria
   */
  static double L2_norm(const dealii::Vector< double >& input_vector, 
                        const dealii::Triangulation< 2 >& tria);
};

#endif //end of comparing_h