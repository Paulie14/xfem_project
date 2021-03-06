
//output
#include <fstream>
#include <iostream>

#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/persistent_tria.h>

#include "xmodel.hh"
#include "comparing.hh"
#include "adaptive_integration.hh"
#include "xquadrature_cell.hh"

inline void XModel::set_well_band_width_ratio(double band_ratio)
{
    well_band_width_ratio_ = band_ratio;
}

inline double XModel::well_band_width_ratio(void)
{
    return well_band_width_ratio_;
}

/************************************ TEMPLATE IMPLEMENTATION **********************************************/
template<Enrichment_method::Type EnrType>
void XModel::prepare_shape_well_averiges(vector< std::map< unsigned int, double > >& shape_well_averiges, std::vector< XDataCell* > xdata_vec)
{
    MASSERT(n_dofs_ > 0,"Dofs uninitialized yet.");
    
    shape_well_averiges.clear();
    shape_well_averiges.resize(wells.size());
        
    for(unsigned int s = 0; s < xdata_vec.size(); s++)
    {
        XDataCell* xdata = xdata_vec[s];
        
        /*getting dof's indices : 
        * [ FEM(dofs_per_cell), 
        *   n_wells * [SGFEM / XFEM(maximum of n_wells*dofs_per_cell), 
        *              well_dof)]_w 
        * ]
        */
        std::vector<unsigned int> local_dof_indices;
        xdata->get_dof_indices(local_dof_indices, fe.dofs_per_cell); //dof initialization
        
        
        unsigned int n_wells_inside = xdata->n_wells_inside(),  // number of wells with q_points inside the cell
                    n_wells = xdata->n_wells(),                // number of wells affecting the cell
                    dofs_per_cell = fe.dofs_per_cell,
                    n_dofs = xdata->n_dofs();  
        
        if(n_wells_inside == 0) continue; //skip when there is no cross-section with any well
  
  
        DoFHandler<2>::active_cell_iterator cell = xdata->get_cell();
        fe_values.reinit (cell);    //NOTE: only mapping is required
        xdata->map_well_quadrature_points(fe_values.get_mapping());
        
        for(unsigned int w = 0; w < n_wells; w++)   //iterate over all well affecting the cell
        {
            if(xdata->q_points(w).size() == 0)   //skip those with no cross-section
            {
                //TODO: if more wells...
                MASSERT(0,"DO NOT ENTER!\n");
                continue;
            }
            
            Well * well = xdata->get_well(w);
            Quadrature<2> quad (xdata->mapped_q_points(w));
            XFEValues<EnrType> xfevalues(fe,quad, update_values | update_quadrature_points);
            xfevalues.reinit(xdata);

            //dx along the well edge = radius of the well; weights are the same all around
//             double dx = well->circumference() / well->q_points().size();
            double dx = 1.0 / well->q_points().size();
            
            std::vector<double> shape_val_vec(n_dofs,0);
            
            //cycle over quadrature points inside the cell
            //DBGMSG("n_q:%d\n",xdata->q_points(w).size());
            for (unsigned int q=0; q < quad.size(); ++q)
            {
                // filling shape values at first
                for(unsigned int i = 0; i < dofs_per_cell; i++)
                    shape_val_vec[i] += xfevalues.shape_value(i,q)*dx; 
                // filling enrichment shape values
                unsigned int index = dofs_per_cell;
                
                for(unsigned int ww = 0; ww < n_wells; ww++)   //iterate over all well affecting the cell
                for(unsigned int k = 0; k < xdata->n_enriched_dofs(ww); k++)
                {
                    if(xdata->global_enriched_dofs(ww)[k] == 0) continue;  //skip unenriched node
                    
                    double temp = xfevalues.enrichment_value(k,ww,q)*dx;
                    //if(index==7) DBGMSG("s=%d k=%d w=%d index=%d q=%d \t val=%e\n",s,k,w,index,q,temp);
                    shape_val_vec[index] += temp;
                    
                    index++;
                }

                //shape_val_vec[index] += -1.0*dx;  //testing function of the well
            
            } // for q
            
            for(unsigned int index = 0; index < n_dofs; index++)
                shape_well_averiges[xdata->get_well_index(w)][local_dof_indices[index]] += shape_val_vec[index];
        } // for w
    } // for s
    
    //correction of wells integration
    for(unsigned int w = 0; w < wells.size(); w++)   //iterate over all well affecting the cell 
    {
        unsigned int well_dof_index = n_dofs_ - wells.size() + w;
        shape_well_averiges[w][well_dof_index] = -1.0; //*wells[w]->circumference();
    }
}



/**********************************           recursive_output                 ******************************/


template<Enrichment_method::Type EnrType>
int XModel::recursive_output(double tolerance, PersistentTriangulation< 2,2  >& output_grid, 
                             DoFHandler<2> &temp_dof_handler, 
                             FE_Q<2> &temp_fe, 
                             const unsigned int iter,
                             unsigned int m
                            )
{ 
  bool refine = false,
       cell_refined;
  unsigned int vertices_per_cell = GeometryInfo<2>::vertices_per_cell,
               dofs_per_cell = fe.dofs_per_cell,
               n_nodes = output_grid.n_vertices(),
               count_cells = 0;
  double max_diff = 0;
  
  //setting new size and initialize with zeros
  dist_enriched.reinit(n_nodes);
  //dist_solution.reinit(n_nodes);
  
  QGauss<2> temp_quad(3);
  FEValues<2> temp_fe_values(temp_fe,temp_quad, update_values | update_quadrature_points | update_JxW_values);
  ConstraintMatrix temp_hanging_node_constraints;

  DoFTools::make_hanging_node_constraints (temp_dof_handler, temp_hanging_node_constraints);  
  temp_hanging_node_constraints.close();
  
  XDataCell *cell_xdata;
  XFEValues<EnrType> xfevalues(fe,quadrature_formula, UpdateFlags::update_default);
  
  std::vector<unsigned int> temp_local_dof_indices (temp_fe_values.dofs_per_cell);
  std::vector<unsigned int> local_dof_indices (xfevalues.dofs_per_cell);
  
  DoFHandler<2>::active_cell_iterator
    cell = temp_dof_handler.begin_active(),
    endc = temp_dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    //DBGMSG("cell: %d\n",cell->index());
    // is there is NOT a user pointer on the cell
    if (cell->user_pointer() == nullptr)
    {
      //if parent is enriched then provide user pointer to children, else continue(the cell is out of the enriched area) 
      if( (cell->level() != 0) && 
          (cell->parent()->user_pointer() != nullptr) 
        )
      {
        cell->set_user_pointer(cell->parent()->user_pointer());
        cell_refined = true;
        //cell->parent()->recursively_set_user_pointer(cell->parent()->user_pointer());
      }
      else
        continue;
    }
    else
        cell_refined = false;
    //else it is enriched and we must compute the difference
    
    temp_fe_values.reinit(cell);
    cell->get_dof_indices(temp_local_dof_indices);
    
    cell_xdata = static_cast<XDataCell*>(cell->user_pointer());
    xfevalues.reinit(cell_xdata);
    cell_xdata->get_cell()->get_dof_indices(local_dof_indices);
    
    //enriched and complete solution
    for(unsigned int i=0; i < vertices_per_cell; i++)
    {
      dist_enriched[temp_local_dof_indices[i]] = 0;
      for(unsigned int w=0; w < cell_xdata->n_wells(); w++)
      {
        // hack for cells inside the well
        if(cell_xdata->get_well(w)->points_inside(cell->vertex(i)))
        {
            dist_unenriched[temp_local_dof_indices[i]] = cell_xdata->get_well(w)->pressure();
            dist_solution[temp_local_dof_indices[i]] = cell_xdata->get_well(w)->pressure();
            break;
        }
        for(unsigned int j=0; j < dofs_per_cell; j++)
        {
          if(cell_xdata->global_enriched_dofs(w)[j] == 0) continue;  //skip unenriched node_enrich_value
          
          dist_enriched[temp_local_dof_indices[i]] += block_solution.block(m)(cell_xdata->global_enriched_dofs(w)[j]) *
                                                      xfevalues.enrichment_value(j,w,cell->vertex(i));
        }
      }
      dist_solution[temp_local_dof_indices[i]] = dist_enriched[temp_local_dof_indices[i]] + dist_unenriched[temp_local_dof_indices[i]];
    }
    
    if(iter > 0)
    {
    if( (! cell_refined) && (iter != 0))
    {
        //DBGMSG("continue on cell %d level %d\n",cell->index(), cell->level());
        continue;
    }
    
    double difference = 0,
           integral = 0;
           
    //DBGMSG("n_q: %d\n",temp_fe_values.n_quadrature_points);
    //Integrate difference between interpolation and xfem solution
    for(unsigned int q=0; q < temp_fe_values.n_quadrature_points; q++)
    {
      // hack for cells inside the well
      bool hack = false;
      for(unsigned int w=0; w < cell_xdata->n_wells(); w++)
      {
        if(cell_xdata->get_well(w)->points_inside(temp_fe_values.get_quadrature_points()[q]))
          hack = true;
      }
      if(hack) continue;
      
      Point<2> unit_point = xfevalues.get_mapping().transform_real_to_unit_cell(cell_xdata->get_cell(),
                                                                                temp_fe_values.get_quadrature_points()[q]);
      double inter = 0,
             sol = 0; 
      for(unsigned int i=0; i < dofs_per_cell; i++)
      {
        inter += dist_solution[temp_local_dof_indices[i]] * temp_fe_values.shape_value(i,q);
        sol += block_solution.block(m)(local_dof_indices[i]) * fe.shape_value(i,unit_point);
      }
      for(unsigned int w=0; w < cell_xdata->n_wells(); w++)
      {
        for(unsigned int j=0; j < dofs_per_cell; j++)
        {
          if(cell_xdata->global_enriched_dofs(w)[j] == 0) continue;  //skip unenriched node_enrich_value
          sol += block_solution.block(m)(cell_xdata->global_enriched_dofs(w)[j]) *
                 xfevalues.enrichment_value(j,w,temp_fe_values.quadrature_point(q));
        }
      }
      //DBGMSG("q: %d\t inter: %e\t sol: %f\t jxw: %e\n",q,inter,sol,temp_fe_values.JxW(q));
      difference += std::abs(inter - sol) * temp_fe_values.JxW(q);
      integral += temp_fe_values.JxW(q);//std::abs(sol) * temp_fe_values.JxW(q);
    }
    
    //DBGMSG("difference: %e, integral: %e, relative: %e, cell: %d, lev: %d\n",difference, integral, difference/integral,cell->index(), cell->level());
    difference = difference / integral; //relative
    if( difference > tolerance)
    {
      count_cells++;
      max_diff = std::max(max_diff, difference);
      cell->set_refine_flag();
      refine = true;
    }
    }
  } // for cells
  
//   block_solution.block(0).print(cout);
//   DBGMSG("size = %d\n",block_solution.block(0).size());
//   dist_enriched.print(cout);
  
    if(iter == 0)
    {
        refine = true;
        output_grid.set_all_refine_flags();
        count_cells = output_grid.n_active_cells();
    }
  DBGMSG("max_diff: %e\t\tcells_for_refinement: %d\n",max_diff, count_cells);
  
  if( (!refine) || (iter == 0) )
    {
    DataOut<2> data_out;
    data_out.attach_dof_handler (temp_dof_handler);
    
    temp_hanging_node_constraints.distribute(dist_unenriched);
    temp_hanging_node_constraints.distribute(dist_enriched);
    temp_hanging_node_constraints.distribute(dist_solution);
    
    if(output_options_ & OutputOptions::output_decomposed)
    {
      data_out.add_data_vector (dist_unenriched, "xfem_unenriched");
      data_out.add_data_vector (dist_enriched, "xfem_enriched"); 
    }
    data_out.add_data_vector (dist_solution, "xfem_solution");
  
    data_out.build_patches ();

    std::stringstream filename;
    filename << output_dir_ << "xmodel_sol_aq" << m << "_" << cycle_;
    if(iter == 0)  filename << "_s";
    filename << ".vtk"; 
   
    std::ofstream output (filename.str());
    data_out.write_vtk (output);

    std::cout << "\nXFEM solution written in:\t" << filename.str() << std::endl;
  }
  
  if(refine)
  {
    output_grid.execute_coarsening_and_refinement();
  // DBGMSG("new n_vertices: %d\n",output_grid.n_vertices());
    temp_dof_handler.distribute_dofs(temp_fe);
    
    dist_unenriched.reinit(temp_dof_handler.n_dofs());
    dist_solution.reinit(temp_dof_handler.n_dofs());
    temp_hanging_node_constraints.clear();
    DoFTools::make_hanging_node_constraints (temp_dof_handler, temp_hanging_node_constraints);  
    temp_hanging_node_constraints.close();
  
    DBGMSG("dofn1: %d\t dofn2: %d\t\n",dof_handler->n_dofs(), temp_dof_handler.n_dofs());
  //  DBGMSG("n1: %d\t n2: %d\t\n",block_solution.block(0).size(), dist_unenriched.size());
    Vector<double>::iterator first = block_solution.block(m).begin();
    Vector<double>::iterator last = first + dof_handler->n_dofs();
  
    VectorTools::interpolate_to_different_mesh(*dof_handler, 
                                             Vector<double>(first, last), 
                                             temp_dof_handler, 
                                             temp_hanging_node_constraints, 
                                             dist_unenriched);
    
    //temp_hanging_node_constraints.distribute(dist_unenriched);
    dist_solution = dist_unenriched;
    
    return 0;
  }
  else
    return 1;

}



/****************************            integrate_difference                 ***********************/

template<Enrichment_method::Type EnrType>
std::pair<double,double> XModel::integrate_difference(dealii::Vector< double >& diff_vector, 
                                                      compare::ExactBase * exact_solution)
{
    unsigned int m = n_aquifers_;
//     unsigned int m = 0;
  
    std::cout << "Computing l2 norm of difference...";
    unsigned int dofs_per_cell = fe.dofs_per_cell,
                 index = 0;
                 
    double exact_value, value, cell_norm, total_norm, nodal_norm, total_nodal_norm;
             
    QGauss<2> temp_quad(3);
    FEValues<2> temp_fe_values(fe,temp_quad, update_values | update_quadrature_points | update_JxW_values);
    std::vector<unsigned int> local_dof_indices (temp_fe_values.dofs_per_cell);   
  
    Vector<double> diff_nodal_vector(dof_handler->n_dofs());
    diff_vector.reinit(dof_handler->get_tria().n_active_cells());
    
    DoFHandler<2>::active_cell_iterator
        cell = dof_handler->begin_active(),
        endc = dof_handler->end();
    for (; cell!=endc; ++cell)
    {
        cell_norm = 0;
        //DBGMSG("cell: %d\n",cell->index());
        // is there is NOT a user pointer on the cell == is not enriched?
        temp_fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        
        if (cell->user_pointer() == nullptr)
        {
            for(unsigned int q=0; q < temp_fe_values.n_quadrature_points; q++)
            {
                value = 0;
                for(unsigned int i=0; i < dofs_per_cell; i++)
                    value += block_solution.block(m)(local_dof_indices[i]) * temp_fe_values.shape_value(i,q);
                
                exact_value = exact_solution->value(temp_fe_values.quadrature_point(q));
                value = value - exact_value;                        // u_h - u
                cell_norm += value * value * temp_fe_values.JxW(q);  // (u_h-u)^2 * JxW
            }
        }
        else
        {            
            XDataCell * xdata = static_cast<XDataCell*>( cell->user_pointer() );
            if( (xdata->n_polar_quadratures() == 0) 
                || 
                ( ! use_polar_quadrature_) )
            {
                XQuadratureCell * xquadrature = new XQuadratureCell(xdata, 
                                                                    fe_values.get_mapping(), 
                                                                    XQuadratureCell::Refinement::edge);
                xquadrature->refine(adaptive_integration_refinement_level_);
                
                Adaptive_integration adaptive_integration(xdata,fe,(XQuadratureBase *)xquadrature,m);
                
                cell_norm += adaptive_integration.integrate_l2_diff<EnrType>(block_solution.block(m),*exact_solution);
            
            } // if
            else
            {
                XQuadratureCell * xquadrature = new XQuadratureCell(xdata, 
                                                                    fe_values.get_mapping(), 
                                                                    XQuadratureCell::Refinement::polar);
                xquadrature->refine(adaptive_integration_refinement_level_);
                    
                AdaptiveIntegrationPolar adaptive_integration_polar(xdata,fe,
                                                                    (XQuadratureBase *)xquadrature,
                                                                    xdata->polar_quadratures(),
                                                                    m);
                
                cell_norm += adaptive_integration_polar.integrate_l2_diff<EnrType>(block_solution.block(m),*exact_solution);
            }
            
            XQuadratureCell * xquadrature = new XQuadratureCell(xdata, 
                                                                fe_values.get_mapping(), 
                                                                XQuadratureCell::Refinement::edge);
            xquadrature->refine(adaptive_integration_refinement_level_);
            
//             //DBGMSG("cell: %d .................callling adaptive_integration.........\n",cell->index());
//             unsigned int t;
//             if(refine_by_error_)    // refinement controlled by tolerance
//             {
//                 for(t=0; t < 17; t++)
//                 {
// //                     DBGMSG("refinement level: %d\n", t);
// //                     if ( ! adaptive_integration.refine_error(alpha_tolerance_))
//                     if ( ! xquadrature->refine_error(alpha_tolerance_))
//                     break;
//                 }
//             }
//             else                    // refinement controlled by edge geometry
//             {
//                 for(t=0; t < adaptive_integration_refinement_level_; t++)
//                 {
// //                     DBGMSG("refinement level: %d\n", t);
// //                     if ( ! adaptive_integration.refine_edge())
// //                     if ( ! adaptive_integration.refine_polar())
//                     
//                     if ( ! xquadrature->refine_edge())
// //                     if ( ! xquadrature->refine_polar())
//                     break;
//                 }
//             }
        }
        
        cell_norm = std::sqrt(cell_norm);   // square root
        diff_vector[index] = cell_norm;     // save L2 norm on cell
        index ++;
        
        //node values should be exactly equal FEM dofs
        for(unsigned int i=0; i < dofs_per_cell; i++)
        {
            nodal_norm = block_solution.block(m)(local_dof_indices[i]) - exact_solution->value(cell->vertex(i));
            diff_nodal_vector[local_dof_indices[i]] = std::abs(nodal_norm);
        }
    }
    
    total_nodal_norm = diff_nodal_vector.l2_norm();
    total_norm = diff_vector.l2_norm();
    std::cout << "\t" << total_norm << "\t vertex l2 norm: " << total_nodal_norm << std::endl;
    
    if(output_options_ & OutputOptions::output_error)
    {
        FE_DGQ<2> temp_fe(0);
        DoFHandler<2>    temp_dof_handler;
        ConstraintMatrix hanging_node_constraints;
  
        temp_dof_handler.initialize(*triangulation,temp_fe);
  
        DoFTools::make_hanging_node_constraints (temp_dof_handler, hanging_node_constraints);  
        hanging_node_constraints.close();
  
        //====================vtk output
        DataOut<2> data_out;
        data_out.attach_dof_handler (temp_dof_handler);
  
        hanging_node_constraints.distribute(diff_vector);
  
        data_out.add_data_vector (diff_vector, "xfem_error");
        data_out.build_patches ();

        std::stringstream filename;
        filename << output_dir_ << "xmodel_error_" << cycle_ << ".vtk";
   
        std::ofstream output (filename.str());
        if(output.is_open())
        {
            data_out.write_vtk (output);
            data_out.clear();
            std::cout << "\noutput(error) written in:\t" << filename.str() << std::endl;
        }
        else
        {
            std::cout << "Could not write the output in file: " << filename.str() << std::endl;
        }
    }
    
    return std::make_pair(total_nodal_norm, total_norm);
}



/****************************            compute_distributed_solution                 ***********************/

template<Enrichment_method::Type EnrType>
void XModel::compute_distributed_solution(const std::vector< Point< 2 > >& points, unsigned int m)
{
  unsigned int n_points = points.size();
  
  //clearing distributed solution vectors
  dist_unenriched.reinit(0);
  dist_enriched.reinit(0);
  dist_solution.reinit(0);
  
  //setting new size and initialize with zeros
  dist_unenriched.reinit(n_points);
  dist_enriched.reinit(n_points);
  dist_solution.reinit(n_points);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  std::pair<DoFHandler<2>::active_cell_iterator, Point<2> > cell_and_point;
  XDataCell *cell_xdata;
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  
  XFEValues<EnrType> xfevalues(fe,quadrature_formula, UpdateFlags::update_default);
  
  Point<2> unit_point;
  unsigned int n=1,
               n_const = points.size() / 10;            //how often we will write DBGMSG
  //iteration over all points where we compute solution
  for (unsigned int p = 0; p < points.size(); p++)
  {
    //only writing DBGMSG to see activity
    if(p == n*n_const)
    {
      DBGMSG("point: p=%d\n",p);
      n++;
    }
      
    //writing zero just for sure
    dist_unenriched[p] = 0;
    dist_enriched[p] = 0;
    dist_solution[p] = 0;
    
    //DBGMSG("point number: %d\n", p);
    //finds cell where points[p] lies and maps that point to unit_point
    //returns pair<cell, unit_point>
    // cell = cell_and_point.first
    // point = cell_and_point.second
    cell_and_point = GridTools::find_active_cell_around_point<2>(fe_values.get_mapping(), *dof_handler, points[p]);

    unit_point = GeometryInfo<2>::project_to_unit_cell(cell_and_point.second); //recommended due to roundoffs
    DoFHandler<2>::active_cell_iterator cell = cell_and_point.first;
    
    if (cell->user_pointer() == nullptr)
    {
        cell->get_dof_indices(local_dof_indices);
        for(unsigned int j=0; j < dofs_per_cell; j++)
        {
            dist_unenriched[p] += block_solution.block(m)(local_dof_indices[j]) 
                                  * fe.shape_value(j, unit_point);
        }
    }
    else
    {
        cell_xdata = static_cast<XDataCell*>(cell->user_pointer());
        xfevalues.reinit(cell_xdata);
        cell_xdata->get_cell()->get_dof_indices(local_dof_indices);
        
        for(unsigned int j=0; j < dofs_per_cell; j++)
        {
            dist_unenriched[p] += block_solution.block(m)(local_dof_indices[j])
                                  * fe.shape_value(j, unit_point);
        }
        
        //enriched solution
        for(unsigned int w=0; w < cell_xdata->n_wells(); w++)
        {
            for(unsigned int j=0; j < dofs_per_cell; j++)
            {
                if(cell_xdata->global_enriched_dofs(w)[j] == 0) continue;  //skip unenriched node_enrich_value
                
                dist_enriched[p] += block_solution.block(m)(cell_xdata->global_enriched_dofs(w)[j]) *
                                                            xfevalues.enrichment_value(j,w,points[p]);
            }
        }
    }

// // OLD code, but clear
//   std::vector<double> local_shape_values (dofs_per_cell);
//   double xshape;
//     //compute shape values (will be used futher down) and unenriched part
//     for(unsigned int j=0; j < dofs_per_cell; j++)
//     {
//       local_shape_values[j] = fe.shape_value(j, unit_point);
//       dist_unenriched[p] += block_solution.block(m)(local_dof_indices[j]) *
//                             local_shape_values[j];
//     }
// 
//     
//     if (cell_and_point.first->user_pointer() != nullptr)
//     {
//       cell_xdata = static_cast<XDataCell*>( cell_and_point.first->user_pointer() );
//       
//       for(unsigned int w = 0; w < cell_xdata->n_wells(); w++)
//       {
//         xshape = cell_xdata->get_well(w)->global_enrich_value(points[p]);
//         double ramp = 0;        
//         double xshape_inter = 0;
//         
//         switch(enrichment_method_)
//         {
//           case Enrichment_method::xfem_shift:
//             //compute value (weight) of the ramp function
//             for(unsigned int l = 0; l < n_vertices; l++)
//             {
//               ramp += cell_xdata->weights(w)[l] * local_shape_values[l];
//             }
//             for(unsigned int k = 0; k < n_vertices; k++)
//             {
//               dist_enriched[p] += block_solution.block(m)(cell_xdata->global_enriched_dofs(w)[k]) *
//                                   ramp *
//                                   local_shape_values[k] *
//                                   (xshape - cell_xdata->get_well(w)->global_enrich_value(cell_and_point.first->vertex(k))); //shifted                      
//             }
//             break;
//             
//           case Enrichment_method::xfem_ramp: 
//             //compute value (weight) of the ramp function
//             for(unsigned int l = 0; l < n_vertices; l++)
//             {
//               ramp += cell_xdata->weights(w)[l] * local_shape_values[l];
//             }
//             for(unsigned int k = 0; k < n_vertices; k++)
//             {
//               dist_enriched[p] += block_solution.block(m)(cell_xdata->global_enriched_dofs(w)[k]) *
//                                   ramp *
//                                   local_shape_values[k] *
//                                   xshape;                      
//             }
//             break;
// 
//           case Enrichment_method::sgfem:
//             //compute value interpolant
//             //DBGMSG("sgfem_compute_solution\n");
//             for(unsigned int l = 0; l < n_vertices; l++) //M_w
//             {
//               //xshape_inter += local_shape_values[l] * cell_xdata->node_enrich_value(w)[l];
//               xshape_inter += local_shape_values[l] * cell_xdata->node_enrich_value(w,l);
//             }
//             for(unsigned int k = 0; k < n_vertices; k++)
//             {
//               if(cell_xdata->global_enriched_dofs(w)[k] != 0)
//               dist_enriched[p] += block_solution.block(m)(cell_xdata->global_enriched_dofs(w)[k]) *
//                                   local_shape_values[k] *
//                                   (xshape - xshape_inter);
//                                   
//               //DBGMSG("dist_enriched[%d] = %e \t\t block=%e \t loc_sh=%e \t\t x_int=%e\n",p,dist_enriched[p],block_solution(cell_xdata->global_enriched_dofs(w)[k]), local_shape_values[k], xshape_inter);
//             }
//             break;
//         } //switch
//         
//       } //for w
//     } //if
//     
     dist_solution[p] = dist_enriched[p] + dist_unenriched[p];
  } //for p
}
