
//output
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

template<Enrichment_method::Type EnrType>
int XModel::recursive_output(double tolerance, PersistentTriangulation< 2  >& output_grid, 
                             DoFHandler<2> &temp_dof_handler, 
                             FE_Q<2> &temp_fe, 
                             const unsigned int cycle)
{ 
  bool refine = false;
  unsigned int vertices_per_cell = GeometryInfo<2>::vertices_per_cell,
               dofs_per_cell = fe.dofs_per_cell,
               n_nodes = output_grid.n_vertices();
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
        cell->parent()->recursively_set_user_pointer(cell->parent()->user_pointer());
      else
        continue;
    }
    
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
      //difference += (inter - sol) * temp_fe_values.JxW(q);
      difference += std::abs(inter - sol) * temp_fe_values.JxW(q);
      integral += std::abs(inter) * temp_fe_values.JxW(q);
    }
    
    //DBGMSG("difference: %e, integral: %e, cell: %d\n",difference, integral,cell->index());
    difference = difference / integral; //relative
    max_diff = std::max(max_diff, difference);
    if( difference > tolerance)
    {
      cell->set_refine_flag();
      refine = true;
    }
  }
  DBGMSG("max_diff: %e\n",max_diff);
    
  if(refine)
  {
    output_grid.execute_coarsening_and_refinement();
    DBGMSG("new n_vertices: %d\n",output_grid.n_vertices());
    temp_dof_handler.distribute_dofs(temp_fe);
    
    /*
    Triangulation<2>::active_cell_iterator
    c = output_grid.begin_active(),
    ec = output_grid.end();
    for (; c!=ec; ++c)
    {
      if( (c->level() != 0) && 
          (c->parent()->user_pointer() != nullptr) 
        )
        c->parent()->recursively_set_user_pointer(c->parent()->user_pointer());
    }
    //*/
    
    dist_unenriched.reinit(temp_dof_handler.n_dofs());
    dist_solution.reinit(temp_dof_handler.n_dofs());
    temp_hanging_node_constraints.clear();
    DoFTools::make_hanging_node_constraints (temp_dof_handler, temp_hanging_node_constraints);  
    temp_hanging_node_constraints.close();
  
    DBGMSG("refining..\n");
    DBGMSG("dofn1: %d\t dofn2: %d\t\n",dof_handler->n_dofs(), temp_dof_handler.n_dofs());
    DBGMSG("n1: %d\t n2: %d\t\n",block_solution.block(0).size(), dist_unenriched.size());
    VectorTools::interpolate_to_different_mesh(*dof_handler, 
                                             block_solution.block(0), 
                                             temp_dof_handler, 
                                             temp_hanging_node_constraints, 
                                             dist_unenriched);
    dist_solution = dist_unenriched;
    return 0;
  }
  else
  {
    DataOut<2> data_out;
    data_out.attach_dof_handler (temp_dof_handler);
    
    temp_hanging_node_constraints.distribute(dist_unenriched);
    temp_hanging_node_constraints.distribute(dist_enriched);
    temp_hanging_node_constraints.distribute(dist_solution);
    
    if(out_decomposed)
    {
      data_out.add_data_vector (dist_unenriched, "xfem_unenriched");
      data_out.add_data_vector (dist_enriched, "xfem_enriched"); 
    }
    data_out.add_data_vector (dist_solution, "xfem_solution");
  
    data_out.build_patches ();

    std::stringstream filename;
    filename << output_dir << "xmodel_sol_" << cycle << ".vtk";
   
    std::ofstream output (filename.str());
    data_out.write_vtk (output);

    std::cout << "\nXFEM solution written in:\t" << filename.str() << std::endl;
    return 1;
  }
}
