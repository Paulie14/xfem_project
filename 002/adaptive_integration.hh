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
#include "xfevalues.hh"

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
    void integrate_xfem( FullMatrix<double> &cell_matrix, 
                         Vector<double> &cell_rhs,
                         std::vector<unsigned int> &local_dof_indices,
                         const double &transmisivity
                       );
    
    
    void integrate_xfem_shift( FullMatrix<double> &cell_matrix, 
                               Vector<double> &cell_rhs,
                               std::vector<unsigned int> &local_dof_indices,
                               const double &transmisivity
                             );
    
    void integrate_xfem_shift2( FullMatrix<double> &cell_matrix, 
                               Vector<double> &cell_rhs,
                               std::vector<unsigned int> &local_dof_indices,
                               const double &transmisivity
                             );
    
//     void integrate_sgfem( FullMatrix<double> &cell_matrix, 
//                           Vector<double> &cell_rhs,
//                           std::vector<unsigned int> &local_dof_indices,
//                           const double &transmisivity
//                         );
    
    void integrate_sgfem2( FullMatrix<double> &cell_matrix, 
                          Vector<double> &cell_rhs,
                          std::vector<unsigned int> &local_dof_indices,
                          const double &transmisivity
                        );
        
    template<Enrichment_method::Type EnrType> 
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
    
    std::vector<Point<2> > q_points_all;
    std::vector<double> jxw_all;
    
    ///Level of current refinement.
    unsigned int level;
    
    ///Does the actual refinement of the squares according to the flags.
    /// @param n_squares_to_refine is number of squares to be refined
    void refine(unsigned int n_squares_to_refine);
    
    void gather_w_points();
  
    ///TODO: Get rid of these
    ///helpful temporary data
    ///mapped well centers to unit cell
    std::vector<Point<2> > m_well_center;
    
    ///mapped well radius to unit cell
    std::vector<double > m_well_radius;
    
};






template<Enrichment_method::Type EnrType> 
void Adaptive_integration::integrate( FullMatrix<double> &cell_matrix, 
                                      Vector<double> &cell_rhs,
                                      std::vector<unsigned int> &local_dof_indices,
                                      const double &transmisivity)
{  
  unsigned int n_wells_inside = 0,                      // number of wells with q_points inside the cell, zero initialized
               n_wells = xdata->n_wells(),              // number of wells affecting the cell
               dofs_per_cell = fe->dofs_per_cell,
               n_vertices = GeometryInfo<2>::vertices_per_cell,
               n_dofs = dofs_per_cell;     //FEM_n_dofs (XFEM_n_dofs_ added later)
               
  QGauss<2> dummy_gauss(1);
  XFEValues<EnrType> xfevalues(*fe,dummy_gauss, update_values | update_quadrature_points | update_covariant_transformation);
  xfevalues.reinit(xdata);
  
  double jacobian = 0, 
         jxw = 0;
         
  //getting unenriched local dofs indices : [FEM(dofs_per_cell), XFEM(n_wells*dofs_per_cell), WELL(n_wells)]
  local_dof_indices.resize(n_dofs);
  cell->get_dof_indices(local_dof_indices);
  
  //getting enriched dof indices and well indices
  for(unsigned int w = 0; w < n_wells; w++)
  {   
    for(unsigned int i = 0; i < n_vertices; i++)
    {
      if(xdata->global_enriched_dofs(w)[i] != 0)
      {
        local_dof_indices.push_back(xdata->global_enriched_dofs(w)[i]);
        n_dofs++;
      }
    }
    if(xdata->q_points(w).size() > 0)
    {
      n_wells_inside++;
      local_dof_indices.push_back(xdata->get_well_dof_index(w)); //one more for well testing funtion
    }
  }
    
  cell_matrix = FullMatrix<double>(n_dofs+n_wells_inside,n_dofs+n_wells_inside);
  cell_matrix = 0;
  cell_rhs = Vector<double>(n_dofs+n_wells_inside);
  cell_rhs = 0;
  
  //vector of quadrature points on the unit square
  std::vector<Point<2> > q_points;
  //vector of quadrature points mapped to unit cell
  std::vector<Point<2> > q_points_mapped;
  
  //temporary vectors for both shape and xshape values and gradients
  std::vector<Tensor<1,2> > shape_grad_vec(n_dofs);
  std::vector<double > shape_val_vec(n_dofs+n_wells_inside,0);
  
  for(unsigned int s=0; s < squares.size(); s++)
  {
    //we are integrating on squares of the unit cell mapped from real cell!!
    q_points = squares[s].gauss->get_points();  //unit square quadrature points
    q_points_mapped = squares[s].gauss->get_points(); 
    squares[s].mapping.map_unit_to_real(q_points_mapped);  //mapped from unit square to unit cell

    Quadrature<2> temp_quad(q_points_mapped);
    FEValues<2> temp_fe_values(*fe,temp_quad, update_values | update_gradients | update_jacobians);
    temp_fe_values.reinit(cell);

    /*
    //testing print of mapped q_points
    if (s == 1)
    {
      DBGMSG("mapping q_points:\n");
      for(unsigned int q=0; q < q_points.size(); q++)
      {
        std::cout << q_points[q] << " | ";
      }
      std::cout << "\n";
      for(unsigned int q=0; q < q_points.size(); q++)
      {
        std::cout << q_points_mapped[q] << " | ";
      }
      std::cout << "\n";
    }
    //*/
    /*
    //in refinement=3 this cell is enriched but does not cross any well
    if (cell->index() == 33)
    {
      DBGMSG("integration:s_jakobian: %f cell_jakobian: %f\n",squares[s].mapping.jakobian(),cell_jakobian);
      DBGMSG("number of wells affecting this cell: %d\n", n_wells);
    }
    //*/
    jacobian = squares[s].mapping.jakobian(); //square.mapping.jakobian = area of the square
    
    for(unsigned int q=0; q < q_points.size(); q++)
    {
      jxw = jacobian * temp_fe_values.jacobian(q).determinant() * squares[s].gauss->get_weights()[q];
          
      // filling FE shape values and shape gradients at first
      for(unsigned int i = 0; i < dofs_per_cell; i++)
      {
        
        shape_grad_vec[i] = temp_fe_values.shape_grad(i,q);
        
#ifdef SOURCES //----------------------------------------------------------------------------sources
        if(n_wells_inside > 0)
          shape_val_vec[i] = temp_fe_values.shape_value(i,q);
#endif
      }

      // filling xshape values and xshape gradients next
      
      unsigned int index = dofs_per_cell; //index in the vector of values and gradients
      for(unsigned int w = 0; w < n_wells; w++) //W
      { 
        for(unsigned int k = 0; k < n_vertices; k++) //M_w
        { 
          if(xdata->global_enriched_dofs(w)[k] == 0) continue;  //skip unenriched node
          
          shape_grad_vec[index] = xfevalues.enrichment_grad(k,w,mapping->transform_unit_to_real_cell(cell, q_points_mapped[q]));
          
#ifdef SOURCES //----------------------------------------------------------------------------sources
            if(n_wells_inside > 0)
            {
              shape_val_vec[index] = xfevalues.enrichment_value(k,w,mapping->transform_unit_to_real_cell(cell, q_points_mapped[q]));
            }
#endif
          
//           for(unsigned int l = 0; l < n_vertices; l++) //M_w
//           {
//             //gradients of shape functions need to be mapped (computed on the unit cell)
//             //scale_to_unit means inverse scaling
//             shape_grad_vec[index] += 
//                        xdata->weights(w)[l] *
//                        ( temp_fe_values.get_fe().shape_grad(l,q_points_mapped[q])
//                          //temp_fe_values.shape_grad(l,q) 
//                        );
//             DBGMSG("vals[%d] = %f   %f\n",l,temp_fe_values.shape_grad(l,q)[0],temp_fe_values.get_fe().shape_grad(l,q_points_mapped[q])[0]);
//             //DBGMSG("vals[%d] = %f   %f\n",l,temp_fe_values.shape_value(l,q),temp_fe_values.get_fe().shape_value(l,q_points_mapped[q]));
//           } //for l
//           
//           //DBGMSG("shape_grad_vec[%d] = %f  %f \t\t %f  %f\n",index,shape_grad_vec[index][0],shape_grad_vec[index][1], prd[0], prd[1]);
          index ++;
        } //for k
        
#ifdef SOURCES //----------------------------------------------------------------------------sources
        //DBGMSG("index=%d\n",index);
        //DBGMSG("shape_val_vec.size=%d\n",shape_val_vec.size());
        if(n_wells_inside > 0)
          shape_val_vec[index] = -1.0;  //testing function of the well
#endif
      } //for w
      
      //filling cell matrix now
      //additions to matrix A,R,S from LAPLACE---------------------------------------------- LAPLACE
      
      for(unsigned int i = 0; i < n_dofs; i++)
        for(unsigned int j = 0; j < n_dofs; j++)
        {
          cell_matrix(i,j) += transmisivity * 
                              shape_grad_vec[i] *
                              shape_grad_vec[j] *
                              jxw;
        }

      //addition from SOURCES--------------------------------------------------------------- SOURCES
#ifdef SOURCES
      for(unsigned int w = 0; w < n_wells; w++) //W
      {
        //this condition tests if the quadrature point lies within the well (testing function omega)
        if(xdata->get_well(w)->points_inside(mapping->transform_unit_to_real_cell(cell, q_points_mapped[q])))
        { 
          for(unsigned int i = 0; i < n_dofs+n_wells_inside; i++)
          {
            for(unsigned int j = 0; j < n_dofs+n_wells_inside; j++)
            {  
              cell_matrix(i,j) += xdata->get_well(w)->perm2aquifer() *
                                  shape_val_vec[i] *
                                  shape_val_vec[j] *
                                  jxw;
                                  
            } //for j
          } //for i
        } //if
      } //for w
#endif

    } //for q 
  } //for s
  
  /*
  std::cout << "cell_matrix" << std::endl;
  cell_matrix.print_formatted(std::cout);
  std::cout << std::endl;
    */
  
  
//------------------------------------------------------------------------------ BOUNDARY INTEGRAL
#ifdef BC_NEWTON //------------------------------------------------------------------------bc_newton
  unsigned int n_w_dofs=0;
  
  for(unsigned int w = 0; w < n_wells; w++)
  {
    if(!(xdata->q_points(w).size() > 0)) continue;
    
    //DBGMSG("well number: %d\n",w);
    Well * well = xdata->get_well(w);
    //jacobian = radius of the well; weights are the same all around
    jxw = 2 * M_PI * well->radius() / well->q_points().size();
      
        
    //how many enriched node on the cell from the well w?
    unsigned int n_enriched_dofs=0;
    for(unsigned int l = 0; l < dofs_per_cell; l++)
    {
      if(xdata->global_enriched_dofs(w)[l] != 0)
      {
        n_enriched_dofs ++;
      }
    }  
//       DBGMSG("Printing node_weights:  [");
//         for(unsigned int a=0; a < node_weights.size(); a++)
//         {
//           std::cout << std::setw(6) << node_weights[a] << "  ";
//         }
//         std::cout << "]" << std::endl;
      
    n_w_dofs = dofs_per_cell+n_enriched_dofs+1;
      
    shape_val_vec.clear();
    shape_val_vec.resize(n_w_dofs,0);  //unenriched, enriched, well
    
    //cycle over quadrature points inside the cell
    for (unsigned int q=0; q < xdata->q_points(w).size(); ++q)
    {
      Point<2> q_point = *(xdata->q_points(w)[q]);
      //transforming the quadrature point to unit cell
      Point<2> unit_point = mapping->transform_real_to_unit_cell(cell, q_point);

      // filling shape values at first
      for(unsigned int l = 0; l < dofs_per_cell; l++)
        shape_val_vec[l] = fe->shape_value(l, unit_point);
        
      // filling xshape values next
      unsigned int index = n_vertices; //index in the vector of values
      for(unsigned int k = 0; k < n_vertices; k++) 
      { 
        if(xdata->global_enriched_dofs(w)[k] != 0)
        {
          shape_val_vec[index] = xfevalues.enrichment_value(k,w,q_point);
                     
          //DBGMSG("shape_val_vec[%d]: %f\n",index, shape_val_vec[index]);
          index ++;
          }
      } //for k
        
      shape_val_vec[index] = -1.0;  //testing function of the well
        
        //printing enriched nodes and dofs
//         DBGMSG("Printing shape_val_vec:  [");
//         for(unsigned int a=0; a < shape_val_vec.size(); a++)
//         {
//           std::cout << std::setw(6) << shape_val_vec[a] << "  ";
//         }
//         std::cout << "]" << std::endl;
        
      for (unsigned int i=0; i < n_w_dofs; ++i)
        for (unsigned int j=0; j < n_w_dofs; ++j)
        {
            cell_matrix(i,j) += ( well->perm2aquifer() *
                                  shape_val_vec[i] *
                                  shape_val_vec[j] *
                                  jxw );
        }
    } //end of iteration over q_points
  } // for w
#endif

//     std::cout << "cell_matrix" << std::endl;
//     cell_matrix.print_formatted(std::cout);
//     std::cout << std::endl;
//     //std::cout << "well_cell_matrix" << std::endl;
//     //well_cell_matrix.print_formatted(std::cout);
//     //std::cout << std::endl;
//     
//     cell_rhs.print(std::cout);
//     std::cout << "--------------------- " << std::endl;  
}


#endif  //Adaptive_integration_h

