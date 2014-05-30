#ifndef Exact_model_h
#define Exact_model_h

#include <deal.II/grid/tria.h>
#include <deal.II/grid/persistent_tria.h>
#include <deal.II/dofs/dof_handler.h>
 

#include <deal.II/lac/vector.h>




#include "model_base.hh"
#include "comparing.hh"

using namespace dealii;

class Well;

/// class ExactModel
/**
 * @brief Exact solution of a model with one circle aquifer and one well. 
 * Well is placed in the center \f$[0,0]\f$ and the radius of the circle is set by user.
 */
class ExactModel
{
  public:
    ///Default constructor.
    ExactModel ();
    
    ///Constructor.
    /**
     * @param exact_solution is function representing exact solution
     */
    ExactModel (ExactBase *exact_solution);
    
    ///Destructor
    ~ExactModel();
    
    ///Returns reference to exact solution vector.
    inline const Vector<double> &get_solution()
    { return solution; }
    
    ///Returns reference to the triangulation on which the exact solution was computed.
    inline const Triangulation<2> &get_triangulation()
    { return *dist_tria; }
    
    ///Outputs exact solution on the circle grid (only refinement flags of grid is needed to create mesh)
    /**
     * @param flag_file is file with refinement flags (for coarse circle grid)
     * @param cycle is not needed
     */
    void output_distributed_solution (const std::string &flag_file,
                                      const unsigned int &cycle=0);
    
    ///Outputs exact solution on given triangulation
    /**
     * @param dist_tria is triangulation on which the exact solution should be computed.
     * @param cycle is not needed
     */
    void output_distributed_solution (const Triangulation<2> &dist_tria,
                                      const unsigned int &cycle=0);

  private:   
    ExactBase *exact_solution;  
      
    ///Triangulation for distributing the solution.
    Triangulation<2> dist_coarse_tria; 
    PersistentTriangulation<2> *dist_tria; 

    Vector<double> solution;
};

#endif  //Exact_model_h