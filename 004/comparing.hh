
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
namespace compare
{
  /** @brief Class representing function of the exact solution.
   * 
   * Base class (interface) for the exact solution.
   * @p value returns value of exact solution in given point.
   */
  class ExactBase : public dealii::Function<2>
    {
      public:
        /** @brief Constructor.
         */
        ExactBase();
        
        ///Returns value of exact soution in given point @p p.
        ///@param p is given point
        ///@param component is set to 0 cause it is a scalar function
        virtual double value (const dealii::Point<2>   &p,
                              const unsigned int  component = 0) const = 0;
                              
        virtual dealii::Tensor<1,2> grad (const dealii::Point<2>   &p,
                             const unsigned int  component = 0) const = 0;    
      protected:    
        unsigned int m_;
    };
    
  /** @brief Class representing function of the exact solution.
   * 
   * We can compute value of analytical solution on a circle area with well source placed in the center.
   * The solution is given by
   *   \f{equation}
   *     u(r) = a \log(\frac{r}{R}), \quad \textrm{kde } a=\frac{P_w}{\log\frac{r_w}{R}}
   *   \f}
   * @p value returns value of exact solution in given point.
   */
  class ExactWellBase : public ExactBase
    {
      public:
        /** @brief Constructor.
         * @param well is pointer to @p Well object 
         * @param radius is radius of the circle area
         */
        
        ExactWellBase(Well *well, double radius, double p_dirichlet = 0);
        
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
    };

    
    class ExactSolutionZero : public ExactBase
    {
    public:
      ExactSolutionZero() : ExactBase()
      {}
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
      dealii::Tensor<1,2> grad (const dealii::Point<2>   &p,
                                const unsigned int  component = 0) const override;   
    };
    
    class ExactSolution : public ExactWellBase
    {
    public:
      ExactSolution(Well *well, double radius, double p_dirichlet = 0) : ExactWellBase(well, radius, p_dirichlet) {}
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
      dealii::Tensor<1,2> grad (const dealii::Point<2>   &p,
                                const unsigned int  component = 0) const override;   
    };
    
    class ExactSolution1 : public ExactWellBase
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
    
    class ExactSolution2 : public ExactWellBase
    {
    public:
      ExactSolution2(Well *well, double radius, double k) : ExactWellBase(well, radius, 0), k_(k) {}
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


    class ExactSolution3 : public ExactWellBase
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
    
    
    /** Exact solution taking into account both sigma permeability (well-aquifer) 
     * and c permeability (aquifer-aquifer).
     * Currenlty the best and working analytical solution for one aquifer-well model.
     */
    class ExactSolution4 : public ExactWellBase
    {
    public:
      ExactSolution4(Well *well, double radius, double k, double amplitude);
        
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
      dealii::Tensor<1,2> grad (const dealii::Point<2>   &p,
                                const unsigned int  component = 0) const override;   
    protected:
        double k_, amplitude_;
        
    friend class Source4;
    };
    
    class Source4 : public ExactSolution4
    {
    public:
      //Source1(Well *well, double radius, double k) : ExactSolution1(well, radius, k) {}
      Source4(ExactSolution4 &ex_sol) 
        : ExactSolution4(ex_sol.well_, ex_sol.radius_, ex_sol.k_, ex_sol.amplitude_) {}
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
    };
    
    
    /** Exact solution taking into account both sigma permeability (well-aquifer) 
     * and c permeability (aquifer-aquifer).
     * Currenlty the best and working analytical solution for one aquifer-well model.
     */
    class ExactSolution5 : public ExactWellBase
    {
    public:
      ExactSolution5(Well *well, double amplitude);
      
      void set_well_parameter(double a);
      
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
      dealii::Tensor<1,2> grad (const dealii::Point<2>   &p,
                                const unsigned int  component = 0) const override;   
    protected:
        double amplitude_;
        
    friend class Source5;
    };
    
    class Source5 : public ExactSolution5
    {
    public:
      Source5(ExactSolution5 &ex_sol) 
        : ExactSolution5(ex_sol.well_, ex_sol.amplitude_) {}
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
    };
    
    
    /** Exact solution taking into account both sigma permeability (well-aquifer) 
     * and c permeability (aquifer-aquifer).
     * Currenlty the best and working analytical solution for one aquifer-well model.
     */
    class ExactSolution6 : public ExactWellBase
    {
    public:
      ExactSolution6(Well *well, double k, double amplitude);
      
      void set_well_parameter(double a);
      
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
      dealii::Tensor<1,2> grad (const dealii::Point<2>   &p,
                                const unsigned int  component = 0) const override;   
    protected:
        double k_, amplitude_;
        
    friend class Source6;
    };
    
    class Source6 : public ExactSolution6
    {
    public:
      Source6(ExactSolution6 &ex_sol) 
        : ExactSolution6(ex_sol.well_, ex_sol.k_, ex_sol.amplitude_) {}
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
    };
    
    
    /** Exact solution taking into account both sigma permeability (well-aquifer) 
     * and c permeability (aquifer-aquifer).
     * Currenlty the best and working analytical solution for one aquifer-well model.
     */
    class ExactSolution7 : public ExactWellBase
    {
    public:
      ExactSolution7(Well *well, double k, double amplitude);
      
      void set_well_parameter(double a);
      
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
      dealii::Tensor<1,2> grad (const dealii::Point<2>   &p,
                                const unsigned int  component = 0) const override;   
    protected:
        double k_, amplitude_;
        
    friend class Source7;
    };
    
    class Source7 : public ExactSolution7
    {
    public:
      Source7(ExactSolution7 &ex_sol) 
        : ExactSolution7(ex_sol.well_, ex_sol.k_, ex_sol.amplitude_) {}
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
    };
    
    
    /** Exact solution taking into account both sigma permeability (well-aquifer) 
     * and c permeability (aquifer-aquifer).
     * Currenlty the best and working analytical solution for one aquifer-well model.
     */
    class ExactSolutionMultiple : public ExactBase
    {
    public:
      ExactSolutionMultiple(double k, double amplitude);
      
      void set_wells(std::vector<Well*> wells, std::vector<double> vec_a, std::vector<double> vec_b);
      
        
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
      dealii::Tensor<1,2> grad (const dealii::Point<2>   &p,
                                const unsigned int  component = 0) const override;   
    protected:
        double k_, amplitude_;
        
        std::vector<double> vec_a, vec_b;
        std::vector<Well*> wells_;
        
    friend class SourceMultiple;
    };
    
    class SourceMultiple : public ExactSolutionMultiple
    {
    public:
      SourceMultiple(double transmisivity, ExactSolutionMultiple &ex_sol);
        
      double value (const dealii::Point<2>   &p,
                    const unsigned int  component = 0) const override; 
    protected:
        double transmisivity_;
    };
    
    
    
/** @brief Class contains helpful methods for comparing models.
 * 
 * Contains methods computing \f$L^2\f$ norm \f$\|\cdot\|_{L^2(\Omega)}\f$ of 
 * difference of two solutions on the given triangulation \f$\|u-v\|_2\f$, 
 * norm of the solution on the given triangulation \f$\|u\|_2\f$.
 * Method @p L2_norm_exact computes \f$\|h\|_2\f$ where \f$h\f$ is the exact solution.
 * Method @p L2_norm_diff computes \f$\|h-u\|_2\f$ where \f$h\f$ is the exact solution
 * and \f$u\f$ is given aproximating solution.
 */

  /// Returns \f$L^2\f$ norm of difference between two dealii vectors.
  /** Return -1 if sizes does not match.
   * @param v1
   * @param v2
   * @param tria
   */
  double L2_norm_diff (const dealii::Vector<double> &v1, 
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
  double L2_norm_diff (const dealii::Vector<double> &input_vector, 
                              const dealii::Triangulation< 2 > &tria, 
                              dealii::Function<2>* exact_solution);

  /// Returns \f$L^2\f$ norm of difference between two dealii vectors.
  /** Return -1 if sizes does not match.
   * @param tria
   * @param well
   * @param area_radius
   */
  double L2_norm_exact (const dealii::Triangulation< 2 > &tria, 
                               dealii::Function<2>* exact_solution);
  
  ///Returns \f$L^2\f$ norm of the vector on the given triangulation
  /**
   * @param input_vector
   * @param tria
   */
  double L2_norm(const dealii::Vector< double >& input_vector, 
                        const dealii::Triangulation< 2 >& tria);

} // compare

#endif //end of comparing_h
