#ifndef Exact_model_h
#define Exact_model_h

#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>

#include "model_base.hh"

// forward declarations
namespace dealii{
    template<int,int> class PersistentTriangulation;
}

class Well;
namespace compare{ class ExactBase; }

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
    ExactModel (compare::ExactBase *exact_solution);
    
    ///Destructor
    ~ExactModel();
    
    ///Returns reference to exact solution vector.
    const dealii::Vector<double> &get_solution();
    
    ///Returns reference to the triangulation on which the exact solution was computed.
    const dealii::Triangulation<2> &get_triangulation();
    
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
    void output_distributed_solution (const dealii::Triangulation<2> &dist_tria,
                                      const unsigned int &cycle=0);

  private:   
    compare::ExactBase *exact_solution;  
      
    ///Triangulation for distributing the solution.
    dealii::Triangulation<2> dist_coarse_tria; 
    dealii::PersistentTriangulation<2> *dist_tria; 

    dealii::Vector<double> solution;
};



/****************************************            Implementation          ********************************/

inline const dealii::Vector<double> & ExactModel::get_solution()
{ return solution; }

inline const dealii::Triangulation<2> & ExactModel::get_triangulation()
{ return *dist_tria; }

#endif  //Exact_model_h