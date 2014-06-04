#include "xfevalues.hh"
#include "system.hh"

template<>
void XFEValues<Enrichment_method::xfem_shift>::prepare()
{
  //ramp function
  double ramp;
  q_ramp_values_.resize(n_wells_);
  
  for(unsigned int w=0; w < n_wells_; w++)
  {
    q_ramp_values_[w].resize(this->n_quadrature_points);
    for(unsigned int q=0; q < this->n_quadrature_points; q++)
    {
      ramp = 0;
      for(unsigned int i=0; i < n_vertices_; i++)
      {
        ramp += this->shape_value(i,q) * xdata_->weights(w)[i];
      }
      q_ramp_values_[w][q] = ramp;
    }
  }
}

template<>
void XFEValues<Enrichment_method::sgfem>::prepare()
{
  //FE interpolation
  double interpolation;
  for(unsigned int w=0; w < n_wells_; w++)
  {
    for(unsigned int q=0; q < this->n_quadrature_points; q++)
    {
      interpolation = 0;
      for(unsigned int i=0; i < n_vertices_; i++)
      {
        interpolation += this->shape_value(i,q) * xdata_->node_enrich_value(w,i);
      }
      q_enrich_values_[w][q] -= interpolation;  //we can edit directly the enrichment value
    }
  }
}


template<>
double XFEValues<Enrichment_method::xfem_shift>::enrichment_value(const unsigned int function_no, const unsigned int w, const unsigned int q)
{ 
  MASSERT(update_quadrature_points && this->get_update_flags(), "'update_quadrature_points' flag was not set!");
  return  this->shape_value(function_no,q) *                                    //FE shape function
          q_ramp_values_[w][q] *                                                //ramp function
          (q_enrich_values_[w][q] - xdata_->node_enrich_value(w,function_no));  //shifted
}

template<>
double XFEValues<Enrichment_method::sgfem>::enrichment_value(const unsigned int function_no, const unsigned int w, const unsigned int q)
{
  MASSERT(update_quadrature_points && this->get_update_flags(), "'update_quadrature_points' flag was not set!");
  MASSERT(xdata_->global_enriched_dofs(w)[function_no] != 0, "Shape function for this node undefined.");
  return  this->shape_value(function_no,q) *    //FE shape function
          q_enrich_values_[w][q];               //already substracted interpolation in prepare()
}


template<>
double XFEValues<Enrichment_method::xfem_shift>::enrichment_value(const unsigned int function_no, const unsigned int w, const Point<2> p)
{
  Point<2> unit_point = this->get_mapping().transform_real_to_unit_cell(cell_,p);
  
  //ramp function
  double ramp = 0;
  for(unsigned int i=0; i < n_vertices_; i++)
    ramp += this->get_fe().shape_value(i,unit_point) * xdata_->weights(w)[i];
    
  return  this->get_fe().shape_value(function_no,unit_point) *
          ramp *
          (xdata_->get_well(w)->global_enrich_value(p) - xdata_->node_enrich_value(w,function_no));

}

template<>
double XFEValues<Enrichment_method::sgfem>::enrichment_value(const unsigned int function_no, const unsigned int w, const Point<2> p)
{
  MASSERT(xdata_->global_enriched_dofs(w)[function_no] != 0, "Shape function for this node undefined.");
  Point<2> unit_point = this->get_mapping().transform_real_to_unit_cell(cell_,p);
  
  //interpolation of enrichment function
  double interpolation = 0;
  for(unsigned int i=0; i < n_vertices_; i++)
    interpolation += this->get_fe().shape_value(i,unit_point) * xdata_->node_enrich_value(w,i);
    
  return  this->get_fe().shape_value(function_no,unit_point) *
          (xdata_->get_well(w)->global_enrich_value(p) - interpolation);
}



template<>
Tensor<1,2> XFEValues<Enrichment_method::xfem_shift>::enrichment_grad(const unsigned int function_no, const unsigned int w, const unsigned int q)
{
  
  double xshape_shifted = q_enrich_values_[w][q] - xdata_->node_enrich_value(w,function_no);
  Tensor<1,2> ramp_grad;
  
  for(unsigned int i=0; i < n_vertices_; i++)
  {
    ramp_grad += shape_grad(i,q) * xdata_->weights(w)[i];
  }
  
  return  shape_value(function_no,q) *
          ( ramp_grad * xshape_shifted 
            + 
            q_ramp_values_[w][q] * xdata_->get_well(w)->global_enrich_grad(this->quadrature_point(q)) 
          )
          +
          shape_grad(function_no,q) *
          q_ramp_values_[w][q] * xshape_shifted
          ;
}


template<>
Tensor<1,2> XFEValues<Enrichment_method::sgfem>::enrichment_grad(const unsigned int function_no, const unsigned int w, const unsigned int q)
{
  MASSERT(xdata_->global_enriched_dofs(w)[function_no] != 0, "Shape grad function for this node undefined.");
  
  //interpolation of enrichment function
  Tensor<1,2> interpolation_grad;       //is initialized with zeros
  for(unsigned int i=0; i < n_vertices_; i++)
  {
    interpolation_grad += this->shape_grad(i,q) * xdata_->node_enrich_value(w,i);
  }
    
  return  this->shape_value(function_no,q) *
          (xdata_->get_well(w)->global_enrich_grad(this->quadrature_point(q)) - interpolation_grad) 
          +
          this->shape_grad(function_no,q) *
          q_enrich_values_[w][q];
          
}




//NOT WORKING
template<>
Tensor<1,2> XFEValues<Enrichment_method::xfem_shift>::enrichment_grad(const unsigned int function_no, const unsigned int w, const Point<2> p)
{
  MASSERT(0, "thit method is not working correctly.");
  Point<2> unit_point = this->get_mapping().transform_real_to_unit_cell(cell_,p);
  
  //ramp function
  double ramp = 0;
  Tensor<1,2> ramp_grad;        //is initialized with zeros

  
  std::vector<Tensor<1,2> > vec_grad (n_vertices_);
  std::vector<Tensor<1,2> > vec_grad_trans(n_vertices_);
  
  for(unsigned int i=0; i < n_vertices_; i++)
  {
      vec_grad[i] = this->get_fe().shape_grad(i,unit_point);
      //DBGMSG("vec_grad[%d][0] = %f   vec_grad[%d][1] = %f\n",i,vec_grad[i][0],vec_grad[i][1]);
  }
  
  //const std::vector<Tensor<1,2> > vec_grad_c (vec_grad);
  //const std::vector<Tensor<1,2> > vec_grad_trans_c(vec_grad_trans);
  
  VectorSlice<const std::vector<Tensor<1,2> > > vec_slice_in(vec_grad);
  VectorSlice< std::vector<Tensor<1,2> > > vec_slice_out(vec_grad_trans);
  
  //this->transform(vec_grad_trans, vec_grad, MappingType::mapping_covariant);
  
  this->mapping->transform(vec_slice_in, vec_slice_out, *(this->mapping_data), MappingType::mapping_covariant);
  
  for(unsigned int i=0; i < n_vertices_; i++)
  {
     // DBGMSG("vec_grad_trans[%d][0] = %f   vec_grad_trans[%d][1] = %f\n",i,vec_grad_trans[i][0],vec_grad_trans[i][1]);
  }
  
  for(unsigned int i=0; i < n_vertices_; i++)
  {
    ramp += this->get_fe().shape_value(i,unit_point) * xdata_->weights(w)[i];
    
//     auto grad = this->get_fe().shape_grad(i,unit_point);
//     
//     std::vector<Tensor<1,2> > vec_grad (1);
//     std::vector<Tensor<1,2> > vec_grad_trans(1);
//         vec_grad[0] = grad;
// 
//     
//     this->transform(vec_grad_trans, vec_grad, MappingType::mapping_covariant);
//     ramp_grad += vec_grad_trans[0] * xdata_->weights(w)[i];
    
    //this->mapping->transform(slice_in, slice_out, *mapping_data, MappingType::mapping_covariant);

    //ramp_grad += this->get_fe().shape_grad(i,unit_point) * xdata_->weights(w)[i];    
    //ramp_grad += vec_grad_trans[i] * xdata_->weights(w)[i];
    ramp_grad += vec_slice_out[i] * xdata_->weights(w)[i];
  }
  
  double xshape_shifted = xdata_->get_well(w)->global_enrich_value(p) - xdata_->node_enrich_value(w,function_no);
    
  return  this->get_fe().shape_value(function_no,unit_point) *
          ( ramp_grad * xshape_shifted 
            + 
            ramp * xdata_->get_well(w)->global_enrich_grad(p) 
          )
          +
          this->get_fe().shape_grad(function_no,unit_point) *
          //vec_grad_trans[function_no] *
          ramp * xshape_shifted
          ;
}

//NOT WORKING
template<>
Tensor<1,2> XFEValues<Enrichment_method::sgfem>::enrichment_grad(const unsigned int function_no, const unsigned int w, const Point<2> p)
{
  MASSERT(0, "thit method is not working correctly.");
  MASSERT(xdata_->global_enriched_dofs(w)[function_no] != 0, "Shape grad function for this node undefined.");
  Point<2> unit_point = this->get_mapping().transform_real_to_unit_cell(cell_,p);
  
  //interpolation of enrichment function
  double interpolation = 0;
  Tensor<1,2> interpolation_grad;       //is initialized with zeros
  for(unsigned int i=0; i < n_vertices_; i++)
  {
    interpolation += this->get_fe().shape_value(i,unit_point) * xdata_->node_enrich_value(w,i);
    interpolation_grad += this->get_fe().shape_grad(i,unit_point) * xdata_->node_enrich_value(w,i);
  }
    
  return  this->get_fe().shape_value(function_no,unit_point) *
          (xdata_->get_well(w)->global_enrich_grad(p) - interpolation_grad) 
          +
          this->get_fe().shape_grad(function_no,unit_point) *
          (xdata_->get_well(w)->global_enrich_value(p) - interpolation);
          
}
