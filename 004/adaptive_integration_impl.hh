#include "adaptive_integration.hh"

#include <deal.II/base/function.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>
#include "xquadrature_base.hh"
#include "xquadrature_well.hh"

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
  xdata_->get_dof_indices(local_dof_indices, fe_->dofs_per_cell); //dof initialization
  
  unsigned int n_wells_inside = xdata_->n_wells_inside(),  // number of wells with q_points inside the cell
               n_wells = xdata_->n_wells(),                // number of wells affecting the cell
               dofs_per_cell = fe_->dofs_per_cell,
               n_vertices = GeometryInfo<2>::vertices_per_cell,
               n_dofs = 0;  
               
  n_dofs = dofs_per_cell + xdata_->n_enriched_dofs();
  
  cell_matrix = FullMatrix<double>(n_dofs+n_wells_inside,n_dofs+n_wells_inside);
  cell_matrix = 0;
  cell_rhs = Vector<double>(n_dofs+n_wells_inside);
  cell_rhs = 0;
   
//     unsigned int ww;
//     if( is_cell_in_well(ww) )
//     {
//         for(unsigned int i = 0; i < n_dofs; i++)
//             cell_matrix(i,i) = 1.0;
//         
//         for(unsigned int i = 0; i < dofs_per_cell; i++)
//             cell_rhs(i) = xdata_->get_well(ww)->pressure();
//         
//         return;
//     }
    
    if(n_wells_inside > 0) 
        std::cout << "Number of quadrature points on cell " 
                  << xdata_->get_cell()->index()
                  << " is "
                  << xquad_->size()
                  << std::endl;

  XFEValues<EnrType> xfevalues(*fe_,*xquad_, 
                                         update_values 
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
  xfevalues.reinit(xdata_);
  n_enrich_quad_points += xquad_->size();
  
  //temporary vectors for both shape and xshape values and gradients
  std::vector<Tensor<1,2> > shape_grad_vec(n_dofs);
  std::vector<double > shape_val_vec(n_dofs+n_wells_inside,0);
  
  for(unsigned int q=0; q<xquad_->size(); q++)
  { 
    //TODO: in XFEValues compute dofs internally and according to 'i' return shape value or xshapevalue
    // filling FE shape values and shape gradients at first
    for(unsigned int i = 0; i < dofs_per_cell; i++)
    {   
      shape_grad_vec[i] = xfevalues.shape_grad(i,q);
        
      //if(n_wells_inside > 0 || rhs_function_)  //with SOURCES
      if(rhs_function_)
          shape_val_vec[i] = xfevalues.shape_value(i,q);
    }

    // filling xshape values and xshape gradients next
    unsigned int index = dofs_per_cell; //index in the vector of values and gradients
    for(unsigned int w = 0; w < n_wells; w++) //W
    {
      for(unsigned int k = 0; k < n_vertices; k++) //M_w
      { 
        if(xdata_->global_enriched_dofs(w)[k] == 0) continue;  //skip unenriched node  

        //if(n_wells_inside > 0 || rhs_function_)    //with SOURCES
        if(rhs_function_)
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
        if(rhs_function_)
        {
            //DBGMSG("sources\n");
          cell_rhs(i) += rhs_function_->value(xfevalues.quadrature_point(q)) * shape_val_vec[i] * xfevalues.JxW(q);
//             if(i >= dofs_per_cell)
//             cell_rhs(i) += rhs_function_->value(xfevalues.quadrature_point(q)) * shape_val_vec[i] * xfevalues.JxW(q);
        }
      }
      //addition from SOURCES--------------------------------------------------------------- SOURCES
#ifdef SOURCES
      for(unsigned int w = 0; w < n_wells; w++) //W
      {
        //this condition tests if the quadrature point lies within the well (testing function omega)
        if(xdata_->get_well(w)->points_inside(mapping->transform_unit_to_real_cell(cell, q_points_all[q])))
        { 
          for(unsigned int i = 0; i < n_dofs+n_wells_inside; i++)
          {
            for(unsigned int j = 0; j < n_dofs+n_wells_inside; j++)
            {  
              cell_matrix(i,j) += xdata_->get_well(w)->perm2aquifer() *
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
}


template<Enrichment_method::Type EnrType> 
double Adaptive_integration::integrate_l2_diff(const Vector<double> &solution, const Function<2> &exact_solution)
{  
    unsigned int n_wells = xdata_->n_wells(),              // number of wells affecting the cell
                 dofs_per_cell = fe_->dofs_per_cell,
                 n_vertices = GeometryInfo<2>::vertices_per_cell;  

    double  value = 0, 
            exact_value = 0, 
            cell_norm = 0;
    std::vector<unsigned int> local_dof_indices;

    xdata_->get_dof_indices(local_dof_indices, dofs_per_cell);

    if(xquad_->size() > 0)
    {
        XFEValues<EnrType> xfevalues(*fe_,*xquad_, update_values 
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
        xfevalues.reinit(xdata_);
    
    
        for(unsigned int q=0; q<xquad_->size(); q++)
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
                if(xdata_->global_enriched_dofs(w)[k] == 0) continue;  //skip unenriched node  
            
                value += solution(xdata_->global_enriched_dofs(w)[k]) * xfevalues.enrichment_value(k,w,q);
            }
            
    //         if(std::abs(value-exact_value) > 1e-15)
    //             DBGMSG("cell: %d \tvalue: %e \t exact: %e \t diff: %e\n",cell->index(),value, exact_value, value-exact_value);
            value = value - exact_value;                   // u_h - u
            cell_norm += value * value * xfevalues.JxW(q);  // (u_h-u)^2 * JxW
        }
    }
    return cell_norm;
}


















template<Enrichment_method::Type EnrType> 
void AdaptiveIntegrationPolar::integrate( FullMatrix<double> &cell_matrix, 
                                      Vector<double> &cell_rhs,
                                      std::vector<unsigned int> &local_dof_indices,
                                      const double &transmisivity)
{  
    MASSERT(polar_xquads_[0] != nullptr, "Undefined polar quadrature.");
    //DBGMSG("Adaptive integration on cell %d. Center: [%f,%f].\n", cell->index(), cell->center()[0], cell->center()[1]);

    /*getting dof's indices : 
    * [ FEM(dofs_per_cell), 
    *   n_wells * [SGFEM / XFEM(maximum of n_wells*dofs_per_cell), 
    *              well_dof)]_w 
    * ]
    */
    xdata_->get_dof_indices(local_dof_indices, fe_->dofs_per_cell); //dof initialization

    DoFHandler<2>::active_cell_iterator cell = xdata_->get_cell();
    
    unsigned int n_wells_inside = xdata_->n_wells_inside(),  // number of wells with q_points inside the cell
                n_wells = xdata_->n_wells(),                // number of wells affecting the cell
                dofs_per_cell = fe_->dofs_per_cell,
                n_vertices = GeometryInfo<2>::vertices_per_cell,
                n_dofs = 0;  
                
    n_dofs = dofs_per_cell + xdata_->n_enriched_dofs();
    
    cell_matrix = FullMatrix<double>(n_dofs+n_wells_inside,n_dofs+n_wells_inside);
    cell_matrix = 0;
    cell_rhs = Vector<double>(n_dofs+n_wells_inside);
    cell_rhs = 0;
              
//     // test if the whole cell is inside the well
//     unsigned int ww;
//     if( is_cell_in_well(ww) )
//     {
//         DBGMSG("adaptive integration:    cell %d in well %d\n", cell->index(), ww);
//         for(unsigned int i = 0; i < n_dofs; i++)
//             cell_matrix(i,i) = 1.0;
//         
//         for(unsigned int i = 0; i < dofs_per_cell; i++)
//             cell_rhs(i) = xdata_->get_well(ww)->pressure();
//         
//         return;
//     }
    
    //temporary vectors for both shape and xshape values and gradients
    std::vector<Tensor<1,2> > shape_grad_vec(n_dofs);
    std::vector<double > shape_val_vec(n_dofs+n_wells_inside,0);
               
    //TODO: more wells !!!!!!!!
    SmoothStep smooth_step(xdata_->get_well(0), polar_xquads_[0]->band_width());
  
    if(xquad_->size() > 0)
    {
        
    XFEValues<EnrType> xfevalues(*fe_,*xquad_, 
                                            update_values 
                                        | update_gradients 
                                        //| update_quadrature_points 
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
    xfevalues.reinit(xdata_);
    n_enrich_quad_points += xquad_->size();
    
    for(unsigned int q=0; q<xquad_->size(); q++)
    { 
        //TODO: in XFEValues compute dofs internally and according to 'i' return shape value or xshapevalue
        // filling FE shape values and shape gradients at first
        for(unsigned int i = 0; i < dofs_per_cell; i++)
        {   
        shape_grad_vec[i] = xfevalues.shape_grad(i,q);
            
        //if(n_wells_inside > 0 || rhs_function_)  //with SOURCES
        if(rhs_function_)
            shape_val_vec[i] = xfevalues.shape_value(i,q);
        }

        // filling xshape values and xshape gradients next
        unsigned int index = dofs_per_cell; //index in the vector of values and gradients
        for(unsigned int w = 0; w < n_wells; w++) //W
        {
        for(unsigned int k = 0; k < n_vertices; k++) //M_w
        { 
            if(xdata_->global_enriched_dofs(w)[k] == 0) continue;  //skip unenriched node  

            //if(n_wells_inside > 0 || rhs_function_)    //with SOURCES
            if(rhs_function_)
            shape_val_vec[index] = xfevalues.enrichment_value(k,w,q);
            
            //shape_grad_vec[index] = xfevalues.enrichment_grad(k,w,mapping->transform_unit_to_real_cell(cell, q_points_mapped[q]));
            shape_grad_vec[index] = xfevalues.enrichment_grad(k,w,q);
            index ++;
        } //for k
        } //for w

        double smooth_step_val = smooth_step.value(xquad_->real_point(q));
        //filling cell matrix now
        //additions to matrix A,R,S from LAPLACE---------------------------------------------- LAPLACE  
        for(unsigned int i = 0; i < n_dofs; i++)
        {
            for(unsigned int j = 0; j < n_dofs; j++)
            {
            cell_matrix(i,j) += transmisivity * 
                                smooth_step_val *
                                shape_grad_vec[i] *
                                shape_grad_vec[j] *
                                xfevalues.JxW(q); //weight of gauss * square_jacobian * cell_jacobian;
            }
            //assembling RHS
            if(rhs_function_)
            {
                //DBGMSG("sources\n");
            cell_rhs(i) += rhs_function_->value(xquad_->real_point(q)) * 
                           smooth_step_val *
                           shape_val_vec[i] * 
                           xfevalues.JxW(q);
            }
        }
    }
    }
    
    MappingQ1<2> mapping;

//     DBGMSG(".................polar quad size %d %d\n",polar_xquads_[0]->size(), polar_xquads_[0]->real_points().size());
    XQuadratureWell polar_xquad; 
    polar_xquads_[0]->create_subquadrature(&polar_xquad, cell, mapping);
    
    n_point_check += polar_xquad.size();
    n_enrich_quad_points += polar_xquad.size();
//     DBGMSG(".................polar quad size %d %d\n",polar_xquad.size(), polar_xquad.real_points().size());

    std::cout << "Number of quadrature points on cell " 
                  << xdata_->get_cell()->index()
                  << " is "
                  << xquad_->size() << " + "
                  << polar_xquad.size() << " = "
                  << xquad_->size() + polar_xquad.size()
                  << std::endl;
                  
    if(polar_xquad.size() > 0)
    {
                  
    XFEValues<EnrType> polar_xfevalues(*fe_,polar_xquad, 
                                            update_values 
                                        | update_gradients 
                                        //| update_quadrature_points 
                                        //| update_covariant_transformation 
                                        //| update_transformation_values 
                                        //| update_transformation_gradients
                                        //| update_boundary_forms 
                                        //| update_cell_normal_vectors 
                                        //| update_JxW_values 
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
    polar_xfevalues.reinit(xdata_);
    
    for(unsigned int q=0; q<polar_xquad.size(); q++)
    { 
        //TODO: in XFEValues compute dofs internally and according to 'i' return shape value or xshapevalue
        // filling FE shape values and shape gradients at first
        for(unsigned int i = 0; i < dofs_per_cell; i++)
        {   
        shape_grad_vec[i] = polar_xfevalues.shape_grad(i,q);
            
        //if(n_wells_inside > 0 || rhs_function_)  //with SOURCES
        if(rhs_function_)
            shape_val_vec[i] = polar_xfevalues.shape_value(i,q);
        }

        // filling xshape values and xshape gradients next
        unsigned int index = dofs_per_cell; //index in the vector of values and gradients
        for(unsigned int w = 0; w < n_wells; w++) //W
        {
        for(unsigned int k = 0; k < n_vertices; k++) //M_w
        { 
            if(xdata_->global_enriched_dofs(w)[k] == 0) continue;  //skip unenriched node  

            //if(n_wells_inside > 0 || rhs_function_)    //with SOURCES
            if(rhs_function_)
            shape_val_vec[index] = polar_xfevalues.enrichment_value(k,w,q);
            
            //shape_grad_vec[index] = xfevalues.enrichment_grad(k,w,mapping->transform_unit_to_real_cell(cell, q_points_mapped[q]));
            shape_grad_vec[index] = polar_xfevalues.enrichment_grad(k,w,q);
            index ++;
        } //for k
        } //for w

        double smooth_step_val = 1.0 - smooth_step.value(polar_xquad.polar_point(q)[0]);
        //filling cell matrix now
        //additions to matrix A,R,S from LAPLACE---------------------------------------------- LAPLACE  
        for(unsigned int i = 0; i < n_dofs; i++)
        {
            for(unsigned int j = 0; j < n_dofs; j++)
            {
            cell_matrix(i,j) += transmisivity * 
                                smooth_step_val *
                                shape_grad_vec[i] *
                                shape_grad_vec[j] *
                                polar_xquad.polar_point(q)[0] *
                                polar_xquad.weight(q);
                                //polar_xfevalues.JxW(q); //weight of gauss * square_jacobian * cell_jacobian;
            }
            //assembling RHS
            if(rhs_function_)
            {
                //DBGMSG("sources\n");
            cell_rhs(i) += rhs_function_->value(polar_xquad.real_point(q)) * 
                           smooth_step_val *
                           shape_val_vec[i] * 
                           polar_xquad.polar_point(q)[0] *
                           polar_xquad.weight(q);
                           //polar_xfevalues.JxW(q);
            }
        }
    } //for q
    } // if
   
    //if(n_wells_inside > 0)
//     {
//         std::cout << "cell_matrix" << std::endl;
//         cell_matrix.print_formatted(std::cout);
//         std::cout << std::endl;
//     }
}


template<Enrichment_method::Type EnrType> 
double AdaptiveIntegrationPolar::integrate_l2_diff(const Vector<double> &solution, const Function<2> &exact_solution)
{  
    unsigned int n_wells = xdata_->n_wells(),              // number of wells affecting the cell
                 dofs_per_cell = fe_->dofs_per_cell,
                 n_vertices = GeometryInfo<2>::vertices_per_cell;  

    double  value = 0, 
            exact_value = 0, 
            cell_norm = 0;
    std::vector<unsigned int> local_dof_indices;

    xdata_->get_dof_indices(local_dof_indices, dofs_per_cell);

    //TODO: more wells !!!!!!!!
    SmoothStep smooth_step(xdata_->get_well(0), polar_xquads_[0]->band_width());
    
    if(xquad_->size() > 0)
    {
        XFEValues<EnrType> xfevalues(*fe_,*xquad_, update_values 
                                        | update_quadrature_points 
                                        | update_JxW_values);
        xfevalues.reinit(xdata_);
    
    
        for(unsigned int q=0; q<xquad_->size(); q++)
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
                if(xdata_->global_enriched_dofs(w)[k] == 0) continue;  //skip unenriched node  
            
                value += solution(xdata_->global_enriched_dofs(w)[k]) * xfevalues.enrichment_value(k,w,q);
            }
            
    //         if(std::abs(value-exact_value) > 1e-15)
    //             DBGMSG("cell: %d \tvalue: %e \t exact: %e \t diff: %e\n",cell->index(),value, exact_value, value-exact_value);
            double smooth_step_val = smooth_step.value(xquad_->real_point(q));
            value = smooth_step_val * (value - exact_value);      // mi*(u_h - u)
            cell_norm += value * value * xfevalues.JxW(q);  // mi^2 * (u_h-u)^2 * JxW
        }
    }
    
    MappingQ1<2> mapping;
    XQuadratureWell polar_xquad; 
    polar_xquads_[0]->create_subquadrature(&polar_xquad, xdata_->get_cell(), mapping);
       
    if(polar_xquad.size() > 0)
    {
        XFEValues<EnrType> polar_xfevalues(*fe_,polar_xquad, 
                                                update_values);
        polar_xfevalues.reinit(xdata_);
        for(unsigned int q=0; q<polar_xquad.size(); q++)
        { 
            exact_value = exact_solution.value(polar_xquad.real_point(q));
            value = 0;  
            // unenriched solution
            for(unsigned int i=0; i < dofs_per_cell; i++)
                value += solution(local_dof_indices[i]) * polar_xfevalues.shape_value(i,q);
                    
            // enriched solution
            for(unsigned int w = 0; w < n_wells; w++) //W
            for(unsigned int k = 0; k < n_vertices; k++) //M_w
            { 
                if(xdata_->global_enriched_dofs(w)[k] == 0) continue;  //skip unenriched node  
                
                value += solution(xdata_->global_enriched_dofs(w)[k]) * polar_xfevalues.enrichment_value(k,w,q);
            }
            
        //         if(std::abs(value-exact_value) > 1e-15)
        //             DBGMSG("cell: %d \tvalue: %e \t exact: %e \t diff: %e\n",cell->index(),value, exact_value, value-exact_value);
            double smooth_step_val = 1 - smooth_step.value(polar_xquad.polar_point(q)[0]);
            value = smooth_step_val * (value - exact_value);      // mi*(u_h - u)
            cell_norm += value * value *
                         polar_xquad.polar_point(q)[0] *
                         polar_xquad.weight(q);;  // mi^2 * (u_h-u)^2 * r * W
        }
    }
    
    return cell_norm;
}
