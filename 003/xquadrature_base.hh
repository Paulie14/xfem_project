#ifndef XQUADRATURE_BASE_H
#define XQUADRATURE_BASE_H


#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>

#include "mapping.hh"

//forward declaration
namespace dealii{
        template<int,int> class Mapping;
}
template<int dim,int spacedim=dim> using DealMapping = dealii::Mapping<dim,spacedim>;

class XQuadratureWell;



/** @brief Class representing squares of adaptive refinement of the reference cell.
 * 
 * This class owns vertices of the squares, mapping procedures between square and unit cell.
 */
class Square
{
public: 
    ///@brief Constructor.
    ///@param p1 is down left vertex of the square
    ///@param p2 is up right vertex of the square
    Square(const Point<2> &p1, const Point<2> &p2);
 
    ///@name Getters
    //@{
        double real_diameter() const;               ///< Returns diameter in real coordinates.
        double unit_diameter() const;               ///< Returns diameter in unit cell coordinates.
        dealii::Quadrature<2> const* quadrature() const;    ///< Returns square quadrature.
        
        dealii::Point<2> real_vertex(unsigned int i) const; ///< Returns @p i vertex in real coordinates.
        const dealii::Point<2>*  real_vertices() const;     ///< Returns vertices in real coordinates.
        
        dealii::Point<2> vertex(unsigned int i) const;      ///< Returns @p i vertex in unit cell coordinates.
        const dealii::Point<2>* vertices() const;           ///< Returns vertices in unit coordinates.
    //@}
    
    /// Transforms the square into the real coordinates.
    void transform_to_real_space(const dealii::DoFHandler<2>::active_cell_iterator& cell,
                                 const DealMapping<2> &mapping);
  
    ///Object mappping data between the adaptively created square and unit cell
    MyMapping mapping;
  
    ///Refine flag is set true, if this square should be refined during next refinement run.
    bool refine_flag;
  
    /// Flag is true if the square has already been processed.
    bool processed;
  
    ///Pointer to Gauss quadrature, that owns the quadrature points and their weights.
    dealii::QGauss<2> const *gauss;
  
private:
      /** Vertices of the square.
    *
    * Numbering of the squares:
    * (is different from DealII)
    *       2
    *   3-------2
    *   |       |
    * 3 |       | 1
    *   |       |
    *   0-------1
    *       0
    */
    dealii::Point<2> vertices_[4];
    
    dealii::Point<2> real_vertices_[4];
    /// Length of diagonal in real space.
    double real_diameter_;
    /// Length of diagonal in unit cell space.
    double unit_diameter_;
    /// Flag is true if the @p transform_to_real_space method was called.
    bool transformed_to_real_;
    
    friend XQuadratureWell;
};



class XQuadratureBase : public dealii::Quadrature<2>
{
public:
    XQuadratureBase();
    
    /// Getter for current level of refinement
    unsigned int level() const;
    
    /// Creates refinement of a cell -- new quadrature.
    virtual void refine(unsigned int max_level) = 0;
    
    /** @brief Calls gnuplot to create image of refined element.
     * 
      * Also can save the gnuplot script to file.
      * @param output_dir is the directory for output_dir
      * @param real is true then the element is printed in real coordinates
      * @param show is true then the gnuplot utility is started and plots the refinement on the screen
      */ 
    virtual void gnuplot_refinement(const std::string &output_dir, bool real=true, bool show=false) = 0;
    
    /// @name Getters.
    //@{
    /// Getter for square vector.
    const std::vector<Square> & squares() const;
    
    /// Getter for i-th square.
    const Square & square(unsigned int i) const;
    
    /// Getter for vector of quadrature points in real coordinates.
    const std::vector<dealii::Point<2> > & real_points() const;
    
    /// Getter for i-th quadrature point in real coordinates.
    const dealii::Point<2> & real_point(unsigned int i) const;
    //@}
    
protected:
    ///Does the actual refinement of the squares according to the flags.
    /// @param n_squares_to_refine is number of squares to be refined
    void apply_refinement(unsigned int n_squares_to_refine);

    /// Gathers the quadrature points and their weigths from squares into a single vector.
    virtual void gather_weights_points();
    
    /// Level of current refinement.
    unsigned int level_;
    
    /// Vector of refined squares.
    /// square[i][j] -> i-th square and its j-th vertex
    std::vector<Square> squares_;
    
    /// Vector of quadrature points in real coordinates.
    std::vector<dealii::Point<2> > real_points_;
    
    /// Vector of Gauss quadrature of different order.
    static const std::vector<dealii::QGauss<2> > quadratures_;
    
    // Square refinement criteria constant on the cells without well inside.
    static const double square_refinement_criteria_factor_;
};




/********************************           IMPLEMENTATION                  *********************************/
#include "system.hh"

inline double Square::real_diameter() const
{   MASSERT(transformed_to_real_, "The square must be transformed to real cell.");
    return real_diameter_; }

inline double Square::unit_diameter() const
{ return unit_diameter_; }

inline Point<2> Square::real_vertex(unsigned int i) const
{   MASSERT(transformed_to_real_, "The square must be transformed to real cell.");
    return real_vertices_[i]; }

inline const dealii::Point<2>* Square::real_vertices() const
{   MASSERT(transformed_to_real_, "The square must be transformed to real cell.");
    return real_vertices_; }

inline Point<2> Square::vertex(unsigned int i) const
{ return vertices_[i]; }

inline const dealii::Point<2>* Square::vertices() const
{ return vertices_; }

inline Quadrature<2> const* Square::quadrature() const
{ return gauss; }



inline unsigned int XQuadratureBase::level() const
{ return level_; }

inline const std::vector< Square >& XQuadratureBase::squares() const
{ return squares_; }

inline const Square& XQuadratureBase::square(unsigned int i) const
{ MASSERT(i < squares_.size(), "Index 'i' exceeded size of vector of refinement squares.");
  return squares_[i]; }
    
inline const Point< 2 >& XQuadratureBase::real_point(unsigned int i) const
{ MASSERT(i < real_points_.size(), "Index 'i' exceeded size of vector of quadrature points.");
  return real_points_[i]; }


inline const std::vector< Point< 2 > >& XQuadratureBase::real_points() const
{ return real_points_; }


#endif  //XQUADRATURE_BASE_H