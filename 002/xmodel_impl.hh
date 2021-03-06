
//output
#include "xmodel.hh"

#include <fstream>
#include <iostream>
#include "adaptive_integration.hh"

#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/fe_dgq.h>

template<Enrichment_method::Type EnrType>
int XModel::recursive_output(double tolerance, PersistentTriangulation< 2  >& output_grid, 
                             DoFHandler<2> &temp_dof_handler, 
                             FE_Q<2> &temp_fe, 
                             const unsigned int iter)
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
        for(unsigned int j=0; j < dofs_per_cell; j++)
        {
          if(cell_xdata->global_enriched_dofs(w)[j] == 0) continue;  //skip unenriched node_enrich_value
          
          dist_enriched[temp_local_dof_indices[i]] += block_solution(cell_xdata->global_enriched_dofs(w)[j]) *
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
      Point<2> unit_point = xfevalues.get_mapping().transform_real_to_unit_cell(cell_xdata->get_cell(),
                                                                                temp_fe_values.get_quadrature_points()[q]);
      double inter = 0,
             sol = 0; 
      for(unsigned int i=0; i < dofs_per_cell; i++)
      {
        inter += dist_solution[temp_local_dof_indices[i]] * temp_fe_values.shape_value(i,q);
        sol += block_solution(local_dof_indices[i]) * fe.shape_value(i,unit_point);
      }
      for(unsigned int w=0; w < cell_xdata->n_wells(); w++)
      {
        for(unsigned int j=0; j < dofs_per_cell; j++)
        {
          if(cell_xdata->global_enriched_dofs(w)[j] == 0) continue;  //skip unenriched node_enrich_value
          sol += block_solution(cell_xdata->global_enriched_dofs(w)[j]) *
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
  }
  
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
    filename << output_dir_ << "xmodel_sol_" << cycle_;
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
    VectorTools::interpolate_to_different_mesh(*dof_handler, 
                                             block_solution.block(0), 
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




template<Enrichment_method::Type EnrType>
std::pair<double,double> XModel::integrate_difference(dealii::Vector< double >& diff_vector, const Function< 2 >& exact_solution)
{
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
                    value += block_solution(local_dof_indices[i]) * temp_fe_values.shape_value(i,q);
                
                exact_value = exact_solution.value(temp_fe_values.quadrature_point(q));
                value = value - exact_value;                        // u_h - u
                cell_norm += value * value * temp_fe_values.JxW(q);  // (u_h-u)^2 * JxW
            }
        }
        else
        { 
            Adaptive_integration adaptive_integration(cell, fe, temp_fe_values.get_mapping());
            
            //unsigned int refinement_level = 15;
            for(unsigned int t=0; t < adaptive_integration_refinement_level_; t++)
            {
                //if(t>0) DBGMSG("refinement level: %d\n", t);
                if ( ! adaptive_integration.refine_edge())
                break;
                if (t == adaptive_integration_refinement_level_-1)
                {
                    // (output_dir, false, true) must be set to unit coordinates and to show on screen 
                    //adaptive_integration.gnuplot_refinement(output_dir);
                }
            }
            cell_norm = adaptive_integration.integrate_l2_diff<EnrType>(block_solution,exact_solution);
        }
        
        cell_norm = std::sqrt(cell_norm);   // square root
        diff_vector[index] = cell_norm;     // save L2 norm on cell
        index ++;
        
        //node values should be exactly equal FEM dofs
        for(unsigned int i=0; i < dofs_per_cell; i++)
        {
            nodal_norm = block_solution(local_dof_indices[i]) - exact_solution.value(cell->vertex(i));
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

