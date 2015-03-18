#include "xmodel.hh"
#include "well.hh"
#include "data_cell.hh"
#include "adaptive_integration.hh"
#include "system.hh"
#include "xfevalues.hh"
#include "comparing.hh"

#include <deal.II/grid/persistent_tria.h>

using namespace compare;

void XModel::test_method(ExactBase* exact_solution)
{
    // Setup part of run() method
    cycle_++;
    if(cycle_ == 0)
    {
        make_grid();
        //if initial refinement is set
        /*
        if(grid_create == load || grid_create == load_circle)
        for(unsigned int r=0; r < init_refinement; r++)
            refine_grid();*/
    }
    else if (is_adaptive)
        refine_grid();


    if (triangulation_changed == true)
        setup_system();
    
    //XDataCell::initialize_node_values(node_enrich_values, xdata, wells.size());
    
    assemble_system();
    
    // Set the solution - dofs
    
    unsigned int dofs_per_cell = fe.dofs_per_cell;
                 //index = 0;
                 
    XDataCell * local_xdata;
             
    //QGauss<2> temp_quad(3);
    //FEValues<2> temp_fe_values(fe,temp_quad, update_values | update_quadrature_points | update_JxW_values);
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);   
    
    DoFHandler<2>::active_cell_iterator
        cell = dof_handler->begin_active(),
        endc = dof_handler->end();
    for (; cell!=endc; ++cell)
    {
        //DBGMSG("cell: %d\n",cell->index());
        // is there is NOT a user pointer on the cell == is not enriched?
        cell->get_dof_indices(local_dof_indices);
        
        for(unsigned int i=0; i < dofs_per_cell; i++)
        {
            block_solution.block(0)(local_dof_indices[i]) = exact_solution->value(cell->vertex(i));
//             if(cell->index() == 0)
//                 DBGMSG("cell %d, dof %d, value: %f\n",cell->index(), local_dof_indices[i], block_solution(local_dof_indices[i]));
        }
        
        if (cell->user_pointer())
        {   
            local_xdata = static_cast<XDataCell*>( cell->user_pointer() );
            for(unsigned int w = 0; w < local_xdata->n_wells(); w++) //W
            for(unsigned int k = 0; k < dofs_per_cell; k++) //M_w
            { 
                if(local_xdata->global_enriched_dofs(w)[k] != 0)
                    block_solution.block(0)(local_xdata->global_enriched_dofs(w)[k]) = exact_solution->a();
//                 if(cell->index() == 0)
//                     DBGMSG("cell %d, dof %d, value: %f\n",cell->index(), local_xdata->global_enriched_dofs(w)[k], block_solution(local_xdata->global_enriched_dofs(w)[k]));
            }
        }
    }
    //TODO: fix index
    block_solution.block(0)[block_solution.block(0).size()-1] = wells[0]->pressure();
    
    //BlockVector<double> temp_solution = block_solution;
    
    const unsigned int blocks_dimension = 3;
    unsigned int n[blocks_dimension] = 
                      { dof_handler->n_dofs(), //n1-block(0) unenriched dofs
                        n_enriched_dofs_,      //n2-block(1) enriched dofs
                        (unsigned int) wells.size()          //n3-block(2) average pressures on wells
                      };
    BlockVector<double> residuum;
    residuum.reinit(blocks_dimension);
    //reinitialization of residuum
    //(N,fast=false) .. vector is filled with zeros
    for(unsigned int i=0; i < blocks_dimension; i++)
        residuum.block(i).reinit(n[i]);
    residuum.collect_sizes();
    
    // A*x
//     block_matrix[0].vmult(residuum,block_solution);
//     residuum.add(-1,block_system_rhs);
    
    std::stringstream filename;
        filename << output_dir_ << "residuum.txt";
    std::ofstream output (filename.str());
   
    if(output.is_open())
        {
            residuum.print(output,10,true, false);
            std::cout << "\nresiduum written in:\t" << filename.str() << std::endl;
        }
        else
        {
            std::cout << "Could not write the residuum in file: " << filename.str() << std::endl;
        }
    
    
    DataOut<2> data_out;
    data_out.attach_dof_handler (*dof_handler);
    
//     temp_hanging_node_constraints.distribute(dist_unenriched);
//     temp_hanging_node_constraints.distribute(dist_enriched);
//     temp_hanging_node_constraints.distribute(dist_solution);
    
    data_out.add_data_vector (residuum.block(0), "fem_res");
    data_out.add_data_vector (residuum.block(1), "xfem_res");
  
    data_out.build_patches ();

    
    std::stringstream res_filename;
    res_filename << output_dir_ << "residuum.vtk"; 
   
    std::ofstream res_output (res_filename.str());
    if(output.is_open())
        {
            data_out.write_vtk (res_output);
            std::cout << "\noutput(error) written in:\t" << res_filename.str() << std::endl;
        }
    else
        {
            std::cout << "Could not write the output in file: " << res_filename.str() << std::endl;
        }
    
    /*
    if(out_error_)
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
    //*/
}



double XModel::test_adaptive_integration(Function< 2 >* func, unsigned int level, unsigned int pol_degree)
{  
    for(unsigned int i=0; i < xdata_[0].size(); i++)
        delete xdata_[0][i];
    
    xdata_[0].clear();
    make_grid();

    clock_t start, stop;
    last_run_time_ = 0.0;

    // Start timer 
    MASSERT((start = clock())!=-1, "Measure time error.");

    dof_handler->initialize(*triangulation,fe);
    find_enriched_cells(0);
    XDataCell::initialize_node_values(node_enrich_values[0], xdata_[0], wells.size());
  
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    double adaptive_integral = 0;
    DoFHandler<2>::active_cell_iterator
        cell = dof_handler->begin_active(),
        endc = dof_handler->end();
    for (; cell!=endc; ++cell)
    {
        DBGMSG("on cell %d\n",cell->index());
        fe_values.reinit (cell);
        MASSERT(cell->user_pointer() != nullptr, "Not enriched cell.");
  
        
        Adaptive_integration adaptive_integration(cell,fe,fe_values.get_mapping(),0);
      
        unsigned int t;
        for(t=0; t < level; t++)
        {
            
            //DBGMSG("refinement level: %d\n", t);
            if ( ! adaptive_integration.refine_edge())
                break;
        }
        DBGMSG("cell %d - adaptive refinement level %d\n",cell->index(), t);
        
        //adaptive_integration.gnuplot_refinement(output_dir_);
        
        adaptive_integral += adaptive_integration.test_integration(func);
    }
  
    std::cout << setprecision(16) << adaptive_integral << std::endl;
  
    Well well = *wells[0];
    double width = std::abs(down_left[0]-up_right[0]);
    double integral = width*width - well.radius() * well.radius() * M_PI;
    std::cout << setprecision(16) << integral << std::endl;
  
    double rel_error = std::abs(adaptive_integral - integral)/integral;
    
    std::cout << "relative error: " << setprecision(16) 
        << rel_error << std::endl;
    
    // Stop timer 
    stop = clock();
    last_run_time_ = ((double) (stop-start))/CLOCKS_PER_SEC;
    std::cout << "Run time: " << last_run_time_ << " s" << std::endl;
    //*/
    return rel_error;
}


void XModel::test_enr_error()
{  
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
    std::vector<double> node_values(dofs_per_cell);
    std::vector<double> node_grads(dofs_per_cell);

    dealii::Vector< double > diff_vector_l2, diff_vector_h1sem, diff_vector_h1, diff_vector_h1_est, distance_vec;
    diff_vector_l2.reinit(dof_handler->get_tria().n_active_cells());
    diff_vector_h1sem.reinit(dof_handler->get_tria().n_active_cells());
    diff_vector_h1.reinit(dof_handler->get_tria().n_active_cells());
    diff_vector_h1_est.reinit(dof_handler->get_tria().n_active_cells());
    distance_vec.reinit(dof_handler->get_tria().n_active_cells());
    
    const unsigned int n_q_points = 5;
    QGauss<2> quad(n_q_points);
    FEValues<2> temp_fe_values(fe,quad, update_values | update_gradients | update_JxW_values | update_quadrature_points);
    
    unsigned int index = 0;
    DoFHandler<2>::active_cell_iterator
        cell = dof_handler->begin_active(),
        endc = dof_handler->end();
    for (; cell!=endc; ++cell)
    {
         Point<2> wc = wells[0]->center();
         double cell_distance = cell->center().distance(wc);

        if( cell_distance < cell->diameter() )
        {
            double empty_val = 1e-16;
            diff_vector_l2[index] = empty_val;
            diff_vector_h1sem[index] = empty_val;
            diff_vector_h1[index] = empty_val;
            distance_vec[index] = cell_distance;
            diff_vector_h1_est[index] = empty_val;
            index++;
            continue;
        }
        
        DBGMSG("on cell %d\n",cell->index());
        temp_fe_values.reinit (cell);
      
        for (unsigned int i=0; i < dofs_per_cell; ++i)
        {
            double distance = wc.distance(cell->vertex(i));
            if (distance <= wells[0]->radius()) distance = wells[0]->radius();
                
            node_values[i] = std::log(distance);
            node_grads[i] = 1.0/distance;
        }
            
        double int_val = 0,
               int_grad = 0;
        
        for (unsigned int q_point=0; q_point < temp_fe_values.n_quadrature_points; ++q_point)
        {
            Point<2> point = temp_fe_values.quadrature_point(q_point);
            double distance = wc.distance(point);
            if (distance <= wells[0]->radius()) distance = wells[0]->radius();
            //if (distance <= 1e-10) continue;
            double interpolation_val = 0;
                   //interpolation_grad = 0;
            Tensor<1,2> interpolation_grad, grad; 
            
            for (unsigned int i=0; i < dofs_per_cell; ++i)
            {
                interpolation_val += node_values[i] * temp_fe_values.shape_value(i, q_point);
                //interpolation_grad += node_grads[i] * temp_fe_values.shape_value(i, q_point);
                interpolation_grad += node_values[i] * temp_fe_values.shape_grad(i, q_point);
            }
            
            grad[0] = (point[0] - wc[0]) / (distance*distance);
            grad[1] = (point[1] - wc[1]) / (distance*distance);
            int_val += pow(std::log(distance) - interpolation_val, 2) * temp_fe_values.JxW (q_point);
            int_grad += (grad - interpolation_grad).norm_square() * temp_fe_values.JxW (q_point);
        }
        
        diff_vector_l2[index] = sqrt(int_val);
        diff_vector_h1sem[index] = sqrt(int_grad);
        diff_vector_h1[index] = sqrt(int_val + int_grad);
        distance_vec[index] = cell_distance;
        
        double el_size = cell->diameter() / sqrt(2.0);
        double estimate_l2norm = pow(el_size,6)/(120 * pow(cell_distance,4));
        double estimate_h1seminorm = pow(el_size,4)/(6 * pow(cell_distance,4));
        double estimate_h1norm = estimate_h1seminorm;//estimate_l2norm + estimate_h1seminorm;
        diff_vector_h1_est[index] = sqrt(estimate_h1seminorm);
        std::cout << "l2norm = " << sqrt(estimate_l2norm) << "\tcomputed = " << sqrt(int_val) << std::endl;
        std::cout << "h1seminorm = " << sqrt(estimate_h1seminorm) << "\tcomputed = " << sqrt(int_grad) << std::endl;
        std::cout << "h1norm = " << sqrt(estimate_h1norm) << "\tcomputed = " << 
            sqrt(int_val + int_grad) << std::endl;
            
        index++;
    }
    
    if(1)
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
  
        hanging_node_constraints.distribute(diff_vector_l2);
        hanging_node_constraints.distribute(diff_vector_h1sem);
        hanging_node_constraints.distribute(diff_vector_h1);
        hanging_node_constraints.distribute(diff_vector_h1_est);
        hanging_node_constraints.distribute(distance_vec);
  
        data_out.add_data_vector (diff_vector_l2, "error_val");
        data_out.add_data_vector (diff_vector_h1sem, "error_grad");
        data_out.add_data_vector (diff_vector_h1, "error_h1");
        data_out.add_data_vector (diff_vector_h1_est, "est_error_h1");
        data_out.add_data_vector (distance_vec, "distance");
        data_out.build_patches ();

        std::stringstream filename;
        filename << output_dir_ << "enr_error_" << cycle_ << ".vtk";
   
        std::ofstream output (filename.str());
        if(output.is_open())
        {
            data_out.write_vtk (output);
            data_out.clear();
            std::cout << "\nlog error written in:\t" << filename.str() << std::endl;
        }
        else
        {
            std::cout << "Could not write the output in file: " << filename.str() << std::endl;
        }
    }
}