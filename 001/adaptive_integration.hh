#ifndef Adaptive_integration_h
#define Adaptive_integration_h

#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/fe_values.h>

#ifdef DEBUG
 // #define DECOMPOSED_CELL_MATRIX
#endif

//#define SOURCES
#define BC_NEWTON
  
#include "mapping.hh"

using namespace dealii;

class XDataCell;

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
  Point<2> vertices[4];
  
  ///Object mappping data between the adaptively created square and unit cell
  MyMapping mapping;
  
  ///Refine flag is set true, if this square should be refined during next refinement run.
  bool refine_flag;
  
  ///Pointer to Gauss quadrature, that owns the quadrature points and their weights.
  const QGauss<2> *gauss;
  
};


/** @brief class doing adaptive integration (in respect to the boundary of the well) on the cell.
 * 
 *  First, it tests if the cell include the well, then does the refinemnt according to the criterion.
 *  Then computes local matrix.
 */
class Adaptive_integration
{
  public:
    /** @brief Constructor.
     * 
      * @param cell is cell iterator for the cell to be adaptively integrated.
      * @param fe is finite element used in FEM on this cell
      * @param mapping is mapping object that maps real cell to reference cell
      */ 
    Adaptive_integration(const DoFHandler<2>::active_cell_iterator &cell, 
                         const FE_Q<2> &fe,
                         const Mapping<2>& mapping
                        );
    
    ///Getter for current level of refinement
    inline unsigned int get_level()
    {return level;}
    
    
    /// @brief Refinement along the well edge.
    /** If the square is crossed by the well edge
      * it will be refined.
      */
    bool refine_edge();
    
    /// @brief Refinement according to the error at each square.
    /** TODO: suggest error computation and comparing
     *  Sofar not implemented
      */ 
    bool refine_error();
    
    
    /** @brief Integrates over all squares and their quadrature points.
      * 
      * @param cell_matrix is a local matrix of the cell
      * @param cell_rhs is a local rhs of the local matrix
      * @param local_dof_indices is vector of dof indices (both unenriched and enriched)
      * @param transmisivity is transmisivity defined on the cell for the Laplace member of the equation
      */
    void integrate( FullMatrix<double> &cell_matrix, 
                    Vector<double> &cell_rhs,
                    std::vector<unsigned int> &local_dof_indices,
                    const double &transmisivity
                  );
    
    /** @brief Calls gnuplot to create image of refined element.
     * 
      * Also can save the gnuplot script to file.
      * @param output_dir is the directory for output_dir
      * @param real is true then the element is printed in real coordinates
      * @param show is true then the gnuplot utility is started and plots the refinement on the screen
      */ 
    void gnuplot_refinement(const std::string &output_dir, bool real=true, bool show=false);
   
    ///1 point Gauss quadrature with dim=2
    static const QGauss<2> gauss_1;
    ///3 point Gauss quadrature with dim=2
    static const QGauss<2> gauss_3;
    ///4 point Gauss quadrature with dim=2
    static const QGauss<2> gauss_4;
    
    
  private:
    ///Current cell to integrate
    const DoFHandler<2>::active_cell_iterator cell;
    
    ///pointer to XFEM data belonging to the cell
    XDataCell *xdata;
    
    ///Finite element of FEM
    const FE_Q<2>  *fe;
    
    ///mapping from real cell to unit cell
    const Mapping<2> *mapping;
    
    ///TODO: Use only DealII mapping for cell_mapping
    ///mapping data of the cell to unit cell
    MyMapping cell_mapping;
    
    ///Vector of refined squares.
    /// square[i][j] -> i-th square and its j-th vertex
    std::vector<Square> squares;
    
    ///Level of current refinement.
    unsigned int level;
    
    ///Does the actual refinement of the squares according to the flags.
    /// @param n_squares_to_refine is number of squares to be refined
    void refine(unsigned int n_squares_to_refine);
  
    ///TODO: Get rid of these
    ///helpful temporary data
    ///mapped well centers to unit cell
    std::vector<Point<2> > m_well_center;
    
    ///mapped well radius to unit cell
    std::vector<double > m_well_radius;
    
};

#endif  //Adaptive_integration_h


