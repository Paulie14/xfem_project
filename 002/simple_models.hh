#ifndef Simple_models_h
#define Simple_models_h

#include "xmodel.hh"
#include "model.hh"
#include <deal.II/base/function.h>


class Well;

/** @brief Class that represents XFEM model with one well.
 * 
 * Corresponds to the FEM model in class @p Model_simple.
 * 
 * Represents XFEM model with one well.
 * Only one well is present, so we have to define Dirichlet boundary condition.
 * That is done in class @p Dirichlet_pressure.
 * 
 * Special treatment is also taken when refining circle grid
 * to satisfy that no hanging nodes are in the enriched area.
 */
class XModel_simple : public XModel
{
  public:
    /// Default constructor
    XModel_simple ();
    
    /// Constructor
    XModel_simple (Well* wells, 
                   const std::string &name="XFEM_Simple_Model",
                   const unsigned int &n_aquifers=1);
    
    /// Destructor
    virtual ~XModel_simple ();
    

    /**Sets file path to a mesh file. 
     * Grid creation type @p grid_create is set to @p load_circle.
     * @param ref_flags is the path to the file with refinement flags
     * @param center is center of the circle
     * @param radius is the radius of the circle
     */
    inline void set_computational_mesh_circle(const std::string &ref_flags, 
                                              const Point<2> &center, 
                                              const double &radius)
    {
      this->center = center;
      this->radius = radius;
      grid_create = load_circle;
      this->ref_flags_file = ref_flags;
    }

  protected:
    
    ///Virtual method overriden to enable circle grid.
    virtual void make_grid();
    ///Virtual method overriden to enable refining of the circle grid.
    virtual void refine_grid();
    
    ///if Dirichlet condition is wanted then this method will do it
    virtual void assemble_dirichlet();
    
    ///center of the circle area of computation
    Point<2> center;
    ///radius of the circle area of computation
    double radius;
    
    ///path to computational mesh
    std::string ref_flags_file;
};


/** @brief Class that represents FEM model with one well.
 * 
 * Corresponds to the XFEM model in class @p XModel_simple.
 * 
 * Represents model with one well.
 * Only one well is present, so we have to define Dirichlet boundary condition.
 * That is done in class @p Dirichlet_pressure.
 * 
 * No special treatment has tobe taken when refining circle grid like in @p XModel_simple. 
 */

class Model_simple : public Model
{
  public:
    /// Default constructor
    Model_simple ();
    
    /// Constructor
    Model_simple (Well* wells, 
                   const std::string &name="FEM_Simple_Model",
                   const unsigned int &n_aquifers=1);
    
    /// Destructor
    virtual ~Model_simple ();

    //virtual void run (const unsigned int cycle);

  protected:
    
    ///Virtual method overriden.
    ///Refine grid apaptively but leaves boundary elements unchanged 
    ///(important to comparision)
    virtual void refine_grid();
    ///if Dirichlet condition is wanted then this method will do it
    virtual void assemble_dirichlet();
};

#endif //Simple_models_h