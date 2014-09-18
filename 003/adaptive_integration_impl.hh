#include "adaptive_integration.hh"

/************************************ INLINE IMPLEMENTATION **********************************************/

inline double Square::real_diameter() const
{ return real_diameter_; }

inline double Square::unit_diameter() const
{ return unit_diameter_; }

inline Point<2> Square::real_vertex(unsigned int i) const
{ return real_vertices_[i]; }

inline Point<2> Square::vertex(unsigned int i) const
{ return vertices[i]; }

inline Quadrature<2> const* Square::quadrature() const
{ return gauss; }
        
inline unsigned int Adaptive_integration::level()
{ return level_; }

inline void Adaptive_integration::set_functors(Function< 2 >* dirichlet_function, 
                                               Function< 2 >* rhs_function)
{
    this->dirichlet_function = dirichlet_function;
    this->rhs_function = rhs_function;
}
/************************************ TEMPLATE IMPLEMENTATION **********************************************/

template<Enrichment_method::Type EnrType> 
void Adaptive_integration::integrate( FullMatrix<double> &cell_matrix, 
                                      Vector<double> &cell_rhs,
                                      std::vector<unsigned int> &local_dof_indices,
                                      const double &transmisivity)
{  
  //DBGMSG("Adaptive integration on cell %d. Center: [%f,%f].\n", cell->index(), cell->center()[0], cell->center()[1]);

  /*getting dof's indices : 
   * [ FEM(dofs_per_cell), 
   *   n_wells * [SGFEM / XFEM(maximum of n_wells*dofs_per_cell), 
   *              well_dof)]_w 
   * ]
   */
  xdata->get_dof_indices(local_dof_indices, fe->dofs_per_cell); //dof initialization
  
  unsigned int n_wells_inside = xdata->n_wells_inside(),  // number of wells with q_points inside the cell
               n_wells = xdata->n_wells(),                // number of wells affecting the cell
               dofs_per_cell = fe->dofs_per_cell,
               n_vertices = GeometryInfo<2>::vertices_per_cell,
               n_dofs = 0;  
               
  n_dofs = dofs_per_cell + xdata->n_enriched_dofs();
  
  cell_matrix = FullMatrix<double>(n_dofs+n_wells_inside,n_dofs+n_wells_inside);
  cell_matrix = 0;
  cell_rhs = Vector<double>(n_dofs+n_wells_inside);
  cell_rhs = 0;
               
  gather_w_points();    //gathers all quadrature points from squares into one vector and maps them to unit cell
  if(q_points_all.size() == 0) return;  // if no quadrature points return zero matrix and rhs 
                                        // (case when whole cell is inside the well - unprobable)
  Quadrature<2> quad(q_points_all, jxw_all);
  XFEValues<EnrType> xfevalues(*fe,quad, update_values 
                                       | update_gradients 
                                       | update_quadrature_points 
                                       //| update_covariant_transformation 
                                       //| update_transformation_values 
                                       //| update_transformation_gradients
                                       //| update_boundary_forms 
                                       //| update_cell_normal_vectors 
                                       | update_JxW_values 
                                       //| update_normal_vectors
                                       //| update_contravariant_transformation
                                       //| update_q_points
                                       //| update_support_points
                                       //| update_support_jacobians 
                                       //| update_support_inverse_jacobians
                                       //| update_second_derivatives
                                       //| update_hessians
                                       //| update_volume_elements
                                       //| update_jacobians
                                       //| update_jacobian_grads
                                       //| update_inverse_jacobians
                                                 );
  xfevalues.reinit(xdata);
  
  //temporary vectors for both shape and xshape values and gradients
  std::vector<Tensor<1,2> > shape_grad_vec(n_dofs);
  std::vector<double > shape_val_vec(n_dofs+n_wells_inside,0);
  
  for(unsigned int q=0; q<q_points_all.size(); q++)
  { 
    //TODO: in XFEValues compute dofs internally and according to 'i' return shape value or xshapevalue
    // filling FE shape values and shape gradients at first
    for(unsigned int i = 0; i < dofs_per_cell; i++)
    {   
      shape_grad_vec[i] = xfevalues.shape_grad(i,q);
        
      //if(n_wells_inside > 0 || rhs_function)  //with SOURCES
      if(rhs_function)
          shape_val_vec[i] = xfevalues.shape_value(i,q);
    }

    // filling xshape values and xshape gradients next
    unsigned int index = dofs_per_cell; //index in the vector of values and gradients
    for(unsigned int w = 0; w < n_wells; w++) //W
    {
      for(unsigned int k = 0; k < n_vertices; k++) //M_w
      { 
        if(xdata->global_enriched_dofs(w)[k] == 0) continue;  //skip unenriched node  

        //if(n_wells_inside > 0 || rhs_function)    //with SOURCES
        if(rhs_function)
          shape_val_vec[index] = xfevalues.enrichment_value(k,w,q);
          
        //shape_grad_vec[index] = xfevalues.enrichment_grad(k,w,mapping->transform_unit_to_real_cell(cell, q_points_mapped[q]));
        shape_grad_vec[index] = xfevalues.enrichment_grad(k,w,q);
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
      {
        for(unsigned int j = 0; j < n_dofs; j++)
        {
          cell_matrix(i,j) += transmisivity * 
                              shape_grad_vec[i] *
                              shape_grad_vec[j] *
                              xfevalues.JxW(q); //weight of gauss * square_jacobian * cell_jacobian;
        }
        //assembling RHS
        if(rhs_function)
        {
            //DBGMSG("sources\n");
          cell_rhs(i) += rhs_function->value(xfevalues.quadrature_point(q)) * shape_val_vec[i] * xfevalues.JxW(q);
        }
      }
      //addition from SOURCES--------------------------------------------------------------- SOURCES
#ifdef SOURCES
      for(unsigned int w = 0; w < n_wells; w++) //W
      {
        //this condition tests if the quadrature point lies within the well (testing function omega)
        if(xdata->get_well(w)->points_inside(mapping->transform_unit_to_real_cell(cell, q_points_all[q])))
        { 
          for(unsigned int i = 0; i < n_dofs+n_wells_inside; i++)
          {
            for(unsigned int j = 0; j < n_dofs+n_wells_inside; j++)
            {  
              cell_matrix(i,j) += xdata->get_well(w)->perm2aquifer() *
                                  shape_val_vec[i] *
                                  shape_val_vec[j] *
                                  xfevalues.JxW(q);
                                  
            } //for j
          } //for i
        } //if
      } //for w
#endif
  }
  
   
//     if(n_wells_inside > 0)
//     {
//         std::cout << "cell_matrix" << std::endl;
//         cell_matrix.print_formatted(std::cout);
//         std::cout << std::endl;
//     }
  
  //------------------------------------------------------------------------------ BOUNDARY INTEGRAL
#ifdef BC_NEWTON //------------------------------------------------------------------------bc_newton
  FullMatrix<double> well_cell_matrix;
  unsigned int n_w_dofs=0;
  double jxw = 0;
  
  //DBGMSG("error n_w:%d\n",n_wells_inside);  
  for(unsigned int w = 0; w < n_wells_inside; w++)
  {
    if(xdata->q_points(w).size() > 0)
    {
      //TODO : map the points somewhere before (in node enrichment routines..)
      //DBGMSG("Cell (%d), mapping well points.\n",cell->index());
      std::vector<Point<2> > points(xdata->q_points(w).size());
      for (unsigned int p =0; p < points.size(); p++)
      {
        //DBGMSG("well point [%d].\n",p);
        points[p] = mapping->transform_real_to_unit_cell(cell,*(xdata->q_points(w)[p]));
      }
      Quadrature<2> quad2 (points);
      XFEValues<EnrType> xfevalues2(*fe,quad2, update_values | update_quadrature_points);
      xfevalues2.reinit(xdata);
      
      Well * well = xdata->get_well(w);
      //jacobian = radius of the well; weights are the same all around
      jxw = 2 * M_PI * well->radius() / well->q_points().size();
    
      shape_val_vec.clear();
      
      // FEM dofs, XFEM dofs, well dof
      //n_w_dofs = dofs_per_cell+n_enriched_dofs+1;
      n_w_dofs = dofs_per_cell + xdata->n_enriched_dofs(w) + 1;
      
      well_cell_matrix.reinit(n_w_dofs, n_w_dofs);
      shape_val_vec.resize(n_w_dofs,0);  //unenriched, enriched, well
    
      std::vector<double > shape_val_averige(n_w_dofs+n_wells,0);
      
      //cycle over quadrature points inside the cell
      //DBGMSG("n_q:%d\n",xdata->q_points(w).size());
      for (unsigned int q=0; q < xdata->q_points(w).size(); ++q)
      {
        // filling shape values at first
        for(unsigned int i = 0; i < dofs_per_cell; i++)
          shape_val_averige[i] = xfevalues2.shape_value(i,q); 
        // filling enrichment shape values
        unsigned int index = dofs_per_cell; //TODO: will not work with more wells on partially enriched cells
        for(unsigned int k = 0; k < n_vertices; k++)
        {
          if(xdata->global_enriched_dofs(w)[k] == 0) continue;  //skip unenriched node
          
          shape_val_averige[index] = xfevalues2.enrichment_value(k,w,q);
          index++;
        }
        
        shape_val_averige[index] = -1.0;  //testing function of the well
        
//         // filling shape values at first
//         for(unsigned int i = 0; i < dofs_per_cell; i++)
//           shape_val_vec[i] = xfevalues2.shape_value(i,q); 
//         // filling enrichment shape values
//         unsigned int index = dofs_per_cell; //TODO: will not work with more wells on partially enriched cells
//         for(unsigned int k = 0; k < n_vertices; k++)
//         {
//           if(xdata->global_enriched_dofs(w)[k] == 0) continue;  //skip unenriched node
//           
//           shape_val_vec[index] = xfevalues2.enrichment_value(k,w,q);
//           index++;
//         }
//         
//         shape_val_vec[index] = -1.0;  //testing function of the well
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
              well_cell_matrix(i,j) += ( shape_val_averige[i]
                                        * shape_val_averige[j]
                                        );
          }
//               // for debugging
//               well_cell_matrix(i,j) += ( well->perm2aquifer() *
//                                     shape_val_vec[i] *
//                                     shape_val_vec[j] *
//                                     jxw );              
//           }
          //*/
      } //q
      
        well_cell_matrix *= ( well->perm2aquifer(m_-1)
                         / xdata->q_points(w).size()
                         * 2 * M_PI * well->radius()
                        );
        
        for (unsigned int i=0; i < n_w_dofs; ++i)
            for (unsigned int j=0; j < n_w_dofs; ++j)
                cell_matrix(i,j) += well_cell_matrix(i,j);
//       for (unsigned int i=0; i < n_w_dofs; ++i)
//         shape_val_averige[i] = shape_val_averige[i] / xdata->q_points(w).size() / (2 * M_PI * well->radius());
//           
//       for (unsigned int i=0; i < n_w_dofs; ++i)
//           for (unsigned int j=0; j < n_w_dofs; ++j)
//           {
//               cell_matrix(i,j) += ( well->perm2aquifer(m_-1)
//                                     * shape_val_averige[i]
//                                     * shape_val_averige[j] 
//                                     //* 2 * M_PI * well->radius()
//                                    );
//               // for debugging
//               well_cell_matrix(i,j) += ( well->perm2aquifer() *
//                                     shape_val_averige[i] *
//                                     shape_val_averige[j]
//                                     );
//           }
          
              
      
    } //if
  } // for w
#endif
 
//     if(n_wells_inside > 0)
//     {
// //     std::cout << "cell_matrix" << std::endl;
// //     cell_matrix.print_formatted(std::cout);
// //     std::cout << std::endl;
//     std::cout << "well_cell_matrix" << std::endl;
//     well_cell_matrix.print_formatted(std::cout);
//     std::cout << std::endl;
//     
//     cell_rhs.print(std::cout);
//     std::cout << "--------------------- " << std::endl;
//     }
    
}


template<Enrichment_method::Type EnrType> 
double Adaptive_integration::integrate_l2_diff(const Vector<double> &solution, const Function<2> &exact_solution)
{  
    unsigned int n_wells = xdata->n_wells(),              // number of wells affecting the cell
                 dofs_per_cell = fe->dofs_per_cell,
                 n_vertices = GeometryInfo<2>::vertices_per_cell;  

    double  value = 0, 
            exact_value = 0, 
            cell_norm = 0;
    std::vector<unsigned int> local_dof_indices;

    xdata->get_dof_indices(local_dof_indices, dofs_per_cell);

    gather_w_points();    //gathers all quadrature points from squares into one vector and maps them to unit cell
    Quadrature<2> quad(q_points_all, jxw_all);
    XFEValues<EnrType> xfevalues(*fe,quad, update_values 
                                       //| update_gradients 
                                       | update_quadrature_points 
                                       //| update_covariant_transformation 
                                       //| update_transformation_values 
                                       //| update_transformation_gradients
                                       //| update_boundary_forms 
                                       //| update_cell_normal_vectors 
                                       | update_JxW_values 
                                       //| update_normal_vectors
                                       //| update_contravariant_transformation
                                       //| update_q_points
                                       //| update_support_points
                                       //| update_support_jacobians 
                                       //| update_support_inverse_jacobians
                                       //| update_second_derivatives
                                       //| update_hessians
                                       //| update_volume_elements
                                       //| update_jacobians
                                       //| update_jacobian_grads
                                       //| update_inverse_jacobians
                                                 );
    xfevalues.reinit(xdata);
  
  
    for(unsigned int q=0; q<q_points_all.size(); q++)
    { 
        exact_value = exact_solution.value(xfevalues.quadrature_point(q));
        value = 0;  
        // unenriched solution
        for(unsigned int i=0; i < dofs_per_cell; i++)
            value += solution(local_dof_indices[i]) * xfevalues.shape_value(i,q);
                
        // enriched solution
        for(unsigned int w = 0; w < n_wells; w++) //W
        for(unsigned int k = 0; k < n_vertices; k++) //M_w
        { 
            if(xdata->global_enriched_dofs(w)[k] == 0) continue;  //skip unenriched node  
         
            value += solution(xdata->global_enriched_dofs(w)[k]) * xfevalues.enrichment_value(k,w,q);
        }
        
//         if(std::abs(value-exact_value) > 1e-15)
//             DBGMSG("cell: %d \tvalue: %e \t exact: %e \t diff: %e\n",cell->index(),value, exact_value, value-exact_value);
        value = value - exact_value;                   // u_h - u
        cell_norm += value * value * xfevalues.JxW(q);  // (u_h-u)^2 * JxW
    }
    
    return cell_norm;
}