
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/logstream.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/iterative_inverse.h>
//#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/eigen.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/base/function.h>

//for adaptive meshes - hanging nodes must be taken care of
#include <deal.II/lac/constraint_matrix.h>

//input/output of grid
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h> 
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/persistent_tria.h>

//for adaptive refinement
#include <deal.II/grid/grid_refinement.h>
//for estimating error
#include <deal.II/numerics/error_estimator.h>

//output
#include <deal.II/numerics/data_out.h>
#include <boost/concept_check.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>      // std::setprecision

//create directory
#include <dirent.h>
#include <sys/stat.h>

#define _USE_MATH_DEFINES       //we are using M_PI
#include <cmath>
#include <vector>

#include "xmodel.hh"
#include "model_base.hh"
#include "well.hh"
#include "data_cell.hh"
#include "adaptive_integration.hh"
#include "system.hh"
#include "xfevalues.hh"
#include "comparing.hh"
#include "xquadrature_base.hh"
#include "xquadrature_well.hh"

using namespace compare;

XModel::XModel () 
  : ModelBase(),
    enrichment_method_(Enrichment_method::xfem_shift),
    well_computation_(Well_computation::bc_newton),
    rad_enr(0),
    n_enriched_dofs_(0),
    n_standard_dofs_(0),
    n_dofs_(0),
    //dealii fem
    triangulation(nullptr),
    fe (1),
    quadrature_formula(2),
    fe_values (fe, quadrature_formula,
      update_values | update_quadrature_points | update_gradients | update_JxW_values),
    hanging_nodes(true),
    output_triangulation(nullptr)
{
    constructor_init();
}

XModel::XModel (const std::string &name, 
                const unsigned int &n_aquifers) 
:   ModelBase::ModelBase(name, n_aquifers),
    enrichment_method_(Enrichment_method::xfem_shift),
    well_computation_(Well_computation::bc_newton),
    rad_enr(0),
    n_enriched_dofs_(0),
    n_standard_dofs_(0),
    n_dofs_(0),
    //dealii fem
    triangulation(nullptr),
    fe (1),
    quadrature_formula(2),
    fe_values (fe, quadrature_formula,
      update_values | update_quadrature_points | update_gradients | update_JxW_values),
    hanging_nodes(true),
    output_triangulation(nullptr)
    
{
    constructor_init();
}

XModel::XModel (const std::vector<Well*> &wells, 
                const std::string &name, 
                const unsigned int &n_aquifers) 
:   ModelBase::ModelBase(wells, name, n_aquifers),
    enrichment_method_(Enrichment_method::xfem_shift),
    well_computation_(Well_computation::bc_newton),
    rad_enr(0),
    n_enriched_dofs_(0),
    n_standard_dofs_(0),
    n_dofs_(0),
    //dealii fem
    triangulation(nullptr),
    fe (1),
    quadrature_formula(2),
    fe_values (fe, quadrature_formula,
      update_values | update_quadrature_points | update_gradients | update_JxW_values),
    hanging_nodes(true),
    output_triangulation(nullptr)
    
{
    constructor_init();
}

void XModel::constructor_init()
{
    dof_handler = new DoFHandler<2>();
    r_enr.resize(wells.size());
    system_matrix_.initialize(n_aquifers_, n_aquifers_);
    if(name_ == "") name_ = "Default_XFEM_Model";
    r_enr_tolerance_ = 33.5;
    refine_by_error_ = false;
    
    use_polar_quadrature_ = false;
    well_band_width_ratio_ = 2;//0.5*std::sqrt(2);
    polar_refinement_level_ = 6;
}


XModel::~XModel()
{
    tria_pointers_.clear();
    
    for(unsigned int i=0; i < xdata_.size(); i++)   //n_aquifers
        for(unsigned int j=0; j < xdata_[i].size(); j++)    //n_enriched_cells
            delete xdata_[i][j];
    
    for(unsigned int w=0; w < well_xquadratures_.size(); w++)   //n_aquifers    
        delete well_xquadratures_[w];
  
    if(dof_handler != nullptr)
        delete dof_handler;
  
    if(output_triangulation != nullptr)
        delete output_triangulation;
  
    if(triangulation != nullptr)
    {
//     triangulation->clear_user_data();
//     triangulation->clear();
        delete triangulation;
    }
}


/************************************ GETTERS AND SETTERS ************************************************/
const Triangulation< 2 >& XModel::get_triangulation()
{
    return *triangulation;
}

std::pair< unsigned int, unsigned int > XModel::get_number_of_dofs()
{
    return std::make_pair(dof_handler->n_dofs(), n_enriched_dofs_);
}

const Triangulation< 2 >& XModel::get_output_triangulation()
{
    return *output_triangulation;
}

const dealii::Vector< double >& XModel::get_distributed_solution()
{
  MASSERT(dist_solution.size() > 0,"Distributed solution has not been computed yet.");
  return dist_solution; 
}


const dealii::Vector< double >& XModel::get_solution()
{
  MASSERT(dist_solution.size() > 0,"Solution has not been computed yet.");
  return dist_solution; 
}

double XModel::well_pressure(unsigned int w)
{
    MASSERT(block_solution.size() >= w, "Solution not computed.");
    return block_solution.block(0)[block_solution.block(0).size() - wells.size() + w];
}

void XModel::set_enrichment_radius(double r_enr)
{
    this->rad_enr = r_enr;
}

void XModel::set_computational_mesh(string coarse_mesh, string ref_flags)
{
    this->coarse_grid_file = coarse_mesh;
    this->ref_flags_file = ref_flags;
    grid_create = load;
}

void XModel::set_well_computation_type(Well_computation::Type well_computation)
{
    well_computation_ = well_computation;
}

void XModel::set_enrichment_method(Enrichment_method::Type enrichment_method)
{
    enrichment_method_ = enrichment_method;
}

void XModel::set_adaptive_refinement_by_error(double alpha_tolerance)
{
    refine_by_error_ = true;
    alpha_tolerance_ = alpha_tolerance;
}


void XModel::make_grid ()
{
  if(dof_handler != nullptr)
  {
    delete dof_handler;
    dof_handler = new DoFHandler<2>();
  }
 
  if(triangulation != nullptr)
  {
        triangulation->clear();
        delete triangulation;
  }
  
  coarse_tria.clear();
  
  switch (grid_create)
  {
    case load:
    {
        MASSERT(coarse_grid_file != "", "Undefined mesh file!");
        //open filestream with mesh from GMSH
        std::ifstream in;
        GridIn<2> gridin;
        in.open(coarse_grid_file);
        //attaching object of triangulation
        gridin.attach_triangulation(coarse_tria);
        if(in.is_open())
        {
          //reading data from filestream
          gridin.read_msh(in);
        }          
        else
        {
          xprintf(Err, "Could not open coarse grid file: %s", coarse_grid_file.c_str());
        }
        
        triangulation = new PersistentTriangulation<2>(coarse_tria);
        in.close();
        in.clear();
        in.open(ref_flags_file);
        if(in.is_open())
          triangulation->read_flags(in);
        else
        {
          xprintf(Warn, "Could not open refinement flags file: %s\n Ingore this if loading mesh without refinement flag file.\n", 
                  ref_flags_file.c_str());
        }
        //creates actual grid to be available
        triangulation->restore();
        break;
    }
    case load_circle:
    {
        xprintf(Err, "Not implemented.");
    }
    case circle:
    {
      Point<2> temp_center((down_left+up_right)/2);
      double temp_radius =  down_left.distance(up_right) / 2;
      GridGenerator::hyper_ball<2>(coarse_tria,temp_center,temp_radius);
      static const HyperBallBoundary<2> boundary(temp_center,temp_radius);
      coarse_tria.set_boundary(0, boundary);
      //coarse_tria.refine_global (init_refinement);
      break;
    }
    case rect:
    default:
    {
      //square grid
      GridGenerator::hyper_rectangle<2>(coarse_tria, down_left, up_right);
      coarse_tria.refine_global (initial_refinement_); 
    }
  }
  
  if(grid_create != load &&  grid_create != load_circle)
  {
    //creates persistent triangulation
    triangulation = new PersistentTriangulation<2>(coarse_tria);
    //creates actual grid to be available
    triangulation->restore();
    
    //MESH OUTPUT - coarse grid = (refinement flags written in output)
    std::stringstream filename1;
    filename1 << output_dir_ << "coarse_grid.msh";
 
    std::ofstream output (filename1.str());
  
    GridOut grid_out;
    grid_out.write_msh<2> (coarse_tria, output);
    std::cout << "Coarse grid written in file: " << filename1.str() << std::endl;
  }
  
  std::cout << "Number of active cells:       "
            << triangulation->n_active_cells()
            << std::endl;
  std::cout << "Total number of cells: "
            << triangulation->n_cells()
            << std::endl;
  triangulation_changed = true;
}

void XModel::refine_grid()
{
  triangulation->refine_global(1);
  triangulation_changed = true;
  
  std::cout << "Number of active cells:       "
            << triangulation->n_active_cells()
            << std::endl;
  std::cout << "Total number of cells: "
            << triangulation->n_cells()
            << std::endl;
}


void XModel::compute_well_quadratures()
{
    //MASSERT(well_xquadratures_.size() == 0, "Well polar quadrature vector is not empty!");
    for(unsigned int w=0; w < well_xquadratures_.size(); w++)   //n_aquifers    
        delete well_xquadratures_[w];
    well_xquadratures_.clear();
    
    for(auto &well: wells)
    {
        double width = well_band_width_ratio_ * well->radius();
        
        well_xquadratures_.push_back(new XQuadratureWell(well, width));
        
        well_xquadratures_.back()->refine(polar_refinement_level_);
//         DBGMSG("polar quad size %d %d\n",well_xquadratures_.back()->size(), well_xquadratures_.back()->real_points().size());
        if(output_options_ & OutputOptions::output_adaptive_plot)
        {   
            string dir_name = "polar_quad";
            well_xquadratures_.back()->gnuplot_refinement(create_subdirectory(output_dir_, dir_name),
                                                          true, false);
        }
    }
}


//------------------------------------------------------------------------------ FIND ENRICHED CELLS
void XModel::find_enriched_cells(unsigned int m)
{
  DBGMSG("find_enriched_cells: wells_size: %d\n",wells.size());
  MASSERT(wells.size() > 0, "No wells are defined in vector of wells");
  
  r_enr.resize(wells.size());
   
  //iterator over cells
  DoFHandler<2>::active_cell_iterator cell,endc;

  //tells us at which global index new enriched dof should be obtained
  unsigned int n_global_enriched_dofs = dof_handler->n_dofs();
  
  std::vector<unsigned int> local_dof_indices(fe.dofs_per_cell);
  
  //on each cell iteration over all wells
  for (unsigned int w = 0; w < wells.size(); w++)
  {
    //temporary vector of enriched dofs: i-th dof is enriched by dof dof_number[i]
    std::vector<unsigned int> enriched_dof_indices; 
    std::vector<unsigned int> enriched_weights;
    //all are initially zero (zero dof is definitely one of the regular dof, so zero means unenriched)
    enriched_dof_indices.resize(dof_handler->n_dofs(),0); 
    enriched_weights.resize(dof_handler->n_dofs(),0); //0 is for unenriched dof
  
    //TODO: use deal ii search algorithm for point in cell
    //finding the cell in which the center lies
    //adding first quadrature points (circle integration)
    cell = dof_handler->begin_active();
    endc = dof_handler->end();
    for (; cell!=endc; ++cell)
    { 
      if ( cell->point_inside(wells[w]->center()) )
      {
        DBGMSG("enrich well: %d from cell: %d\n", w, cell->index());
        
        //getting r_enr (circle in which dofs(nodes) will be enriched)
        //setting first XData for the cell in which the center of the well lies
        r_enr[w]=wells[w]->radius(); //can be set to well radius or 0
        r_enr[w] = std::max(rad_enr, r_enr[w]);
        
        //check off the minimal enrichment radius (has to include at least one node)
        double dist = std::max(cell->vertex(0).distance(wells[w]->center()), r_enr[w]);
        for(unsigned int i=1; i < fe.dofs_per_cell; i++)
        {
          //assumming that the well radius is smaller than the cell
          //dist = std::min(cell->vertex(i).distance(wells[w]->center()), r_enr[w]);
          dist = std::max(cell->vertex(i).distance(wells[w]->center()), dist);
        }  
        r_enr[w] = std::max(dist, r_enr[w]);
          
        // Automatic choice of the enrichment radius according to the given tolerance:
//         r_enr_tolerance_ = (cycle_ == 0) ? (r_enr_tolerance_/10 ) : (r_enr_tolerance_/4);
//         double enr_radius = 0.25 * pow(cell->diameter(), 1.5) / r_enr_tolerance_;
// //         double h = pow(cell->diameter(),5)*0.025 + 0.25*pow(cell->diameter(),3);
// //         double enr_radius = sqrt(h);
//         DBGMSG("r_enr_tolerance = %e, h = %f, enr_radius = %f\n",r_enr_tolerance_, cell->diameter(), enr_radius);
//          r_enr[w] = enr_radius;//std::min(r_enr[w], enr_radius);
        
        std::cout << "enrichment radius: wanted: " << rad_enr << "  finally set: " << r_enr[w] << std::endl;
//             << "\t newly computed: " << enr_radius << std::endl;
        
        triangulation->clear_user_flags();
        
        if( (enrichment_method_ == Enrichment_method::sgfem)
            || (enrichment_method_ == Enrichment_method::xfem)
        )
          enrich_cell(cell, w, enriched_dof_indices, n_global_enriched_dofs,m);
        else
          enrich_cell_blend(cell, w, enriched_dof_indices, enriched_weights, n_global_enriched_dofs,m);
        
        break;
      }
    } //for cells
  } //for wells
  
  
  // computing enrichment tolerance
  for (unsigned int w = 0; w < wells.size(); w++)
  {
    Well well = *(wells[w]);
    Vector<double> diff_vector(triangulation->n_active_cells());
    QGauss<2> temp_quad(4);
    FEValues<2> temp_fe_values(fe,temp_quad, update_values | update_quadrature_points | update_JxW_values);
    
    cell = dof_handler->begin_active();
    endc = dof_handler->end();
    for (; cell!=endc; ++cell)
    { 
        //integral of s(x)-i(s)(x)
        fe_values.reinit(cell);
        
        //node values of the global enrichment function
        std::vector<double> node_values(GeometryInfo<2>::vertices_per_cell);
        for(unsigned int i=0; i < GeometryInfo<2>::vertices_per_cell; i++)
            node_values[i] = well.global_enrich_value(cell->vertex(i));
        
        double integral = 0;
        for(unsigned int q=0; q < fe_values.n_quadrature_points; q++)
        {
            double interpolation = 0;
            for(unsigned int i=0; i < fe.dofs_per_cell; i++)
                interpolation += fe_values.shape_value(i,q) * node_values[i];
            
            integral += abs(well.global_enrich_value(fe_values.quadrature_point(q)) - interpolation) 
                        * fe_values.JxW(q);
        }
    
        
        //tolerance criterion
        double  distance = cell->center().distance(well.center());
        double error_estimate = 1.0/12 * cell->diameter() * cell->diameter() * cell->diameter() 
                                * cell->diameter() / distance / distance / distance;
//         double error_estimate = M_PI/6 * cell->diameter() * cell->diameter() / distance;
    
//         diff_vector[cell->index()] = integral;
        diff_vector[cell->index()] = error_estimate/integral;
        //diff_vector[cell->index()] = (integral / cell->measure() > 1e-3)? 1:0;
        //(error_estimate > 1e-1)? 1 : 0;
        
        //DBGMSG("error_estimate = %e,\tintegral = %e\n",error_estimate, integral);
    }
        FE_DGQ<2> temp_fe(0);
        DoFHandler<2>    temp_dof_handler;
        //ConstraintMatrix hanging_node_constraints;
  
        temp_dof_handler.initialize(*triangulation,temp_fe);
  
        //DoFTools::make_hanging_node_constraints (temp_dof_handler, hanging_node_constraints);  
        //hanging_node_constraints.close();
  
        //====================vtk output
        DataOut<2> data_out;
        data_out.attach_dof_handler (temp_dof_handler);
  
        //hanging_node_constraints.distribute(diff_vector);
  
        data_out.add_data_vector (diff_vector, "xfem_enrichment");
        data_out.build_patches ();

        std::stringstream filename;
        filename << output_dir_ << "xfem_enrichment_" << cycle_ << "_" << w << ".vtk";
   
        std::ofstream output (filename.str());
        if(output.is_open())
        {
            data_out.write_vtk (output);
            data_out.clear();
            std::cout << "\nenrichment area written in:\t" << filename.str() << std::endl;
        }
        else
        {
            std::cout << "Could not write the output in file: " << filename.str() << std::endl;
        }
  }
  
    n_enriched_dofs_ = n_global_enriched_dofs - dof_handler->n_dofs();
    std::cout << "Number of unenriched dofs: "
        << dof_handler->n_dofs()
        << std::endl;
    std::cout << "Number of enriched dofs: " << n_enriched_dofs_ << std::endl;
    std::cout << "Total number of dofs: " << n_enriched_dofs_+dof_handler->n_dofs() << std::endl;
    //MASSERT(n_enriched_dofs > 1, "Must be solved. Crashes somewhere in Adaptive_integration.");
  
    DBGMSG("Printing xdata (n=%d), number of cells (%d)\n",xdata_[m].size(), triangulation->n_active_cells());
//     print_xdata();
}

void XModel::print_xdata()
{
  //printing enriched nodes and dofs
  for(unsigned int m=0; m < xdata_.size(); m++)
  for(unsigned int i=0; i < xdata_[m].size(); i++)
  {
    std::cout << "(" << setw(5) << xdata_[m][i]->get_cell()->index() << ")  ";
    for(unsigned int xw=0; xw < xdata_[m][i]->n_wells(); xw++)
    {
      int width_well = 2;
      if(xw > 0) width_well = 11; 
      std::cout << setw(width_well) << "w=" << setw(3) << xw << " well center: " << setw(7)
      << xdata_[m][i]->get_well(xw)->center() << "\tglobal_enrich_dofs: [";
      
      for(unsigned int j=0; j < fe.dofs_per_cell; j++)
        std::cout << std::setw(4) << xdata_[m][i]->global_enriched_dofs(xw)[j] << "  ";
      std::cout << "]   [";
      
      for(unsigned int j=0; j < fe.dofs_per_cell; j++)
        std::cout << std::setw(4) << xdata_[m][i]->weights(xw)[j] << "  ";
      std::cout << "]";
      
      std::cout << "  n_qpoints=" <<  xdata_[m][i]->q_points(xw).size() << 
      "  boundary: " << xdata_[m][i]->get_cell()->at_boundary() << std::endl;
    }
  }
}


//-------------------------------------------------------------------------------------- ENRICH CELL
void XModel::enrich_cell_blend (const DoFHandler<2>::active_cell_iterator cell,
                                const unsigned int &well_index,
                                std::vector<unsigned int> &enriched_dof_indices,
                                std::vector<unsigned int> &enriched_weights,
                                unsigned int &n_global_enriched_dofs,
                                unsigned int m)
{
  //std::cout << "CALL enrich_cell on: " << cell->index();
  // if the flag is set = we have been already there, so continue
    if ( cell->user_flag_set() == true)
    {
      //std::cout << "\thave already been at cell\n";
      return;
    }
  
  //sets user flag for the cell in which we have been
  cell->set_user_flag();
  
  //flag is true if no node is to be enriched (is not in the enrichment radius )
  bool cell_not_enriching = true;
  
  //TODO: compute the distance only once
  for(unsigned int i=0; i < fe.dofs_per_cell; i++)
  {
    if(cell->vertex(i).distance(wells[well_index]->center()) <= r_enr[well_index])
    {
      cell_not_enriching = false;
    }
  }
  
  //if there is no previously enriched dof we can return
  if (cell_not_enriching)
  {
    return;
  }
  //else
  
  
  std::vector<unsigned int> local_dof_indices(fe.dofs_per_cell);
  cell->get_dof_indices(local_dof_indices);
  
  //else we continue by enriching dofs
  //std::cout << "\tenriched: ";
  //temporary space for enriched dofs
  std::vector<unsigned int> local_enriched_dofs(fe.dofs_per_cell,0);
  std::vector<unsigned int> local_enriched_node_weights(fe.dofs_per_cell,0);
  
  //not enriching conrner, wrong
//   unsigned int bounds = 0;
//   if(cell->at_boundary())
//     {
//         for (unsigned int face_no=0; face_no < GeometryInfo<2>::faces_per_cell; ++face_no)
//         {
//             //std::cout << "\tface(" << face_no << "): ";
//             //typename DoFHandler<2>::face_iterator face = cell->face(face_no);
//    
//             //if the face is at the boundary, we do not enrich nodes
//             if (cell->at_boundary(face_no)) 
//                 bounds++;
//         }
//     }
//   if(bounds > 1) return;
  /*
    const unsigned int weigth_boundary = 2;
    if(cell->at_boundary())
    {
        for (unsigned int face_no=0; face_no < GeometryInfo<2>::faces_per_cell; ++face_no)
        {
            //std::cout << "\tface(" << face_no << "): ";
            //typename DoFHandler<2>::face_iterator face = cell->face(face_no);
   
            //if the face is at the boundary, we do not enrich nodes
            if (cell->at_boundary(face_no)) 
            {
                //DBGMSG("dofs_per_cell: %d\n", fe.dofs_per_face);
                //DBGMSG("cell: %d, face: %d\n", cell->index(),face_no);
                for(unsigned int i=0; i < GeometryInfo<2>::vertices_per_face; i++)    
                    enriched_weights[cell->face(face_no)->vertex_dof_index(i,0)] = weigth_boundary;
//                 for(unsigned int i=0; i < fe.dofs_per_face; i++)    
//                 enriched_weights[cell->face(face_no)->dof_index(i)] = weigth_boundary;
            }   
        }
    }
  //*/
  
  
  //enriching dofs
  for(unsigned int i=0; i < fe.dofs_per_cell; i++)
  {
//     //if the dof is on the boundary
//     if(enriched_weights[local_dof_indices[i]] == weigth_boundary) 
//     {
//         enriched_weights[local_dof_indices[i]] = 0;
//     }
    //if the node was previously enriched
    if (enriched_dof_indices[local_dof_indices[i]] != 0)
    {
      //adding enriched dof that has been already defined from previous cell
      local_enriched_dofs[i] = enriched_dof_indices[local_dof_indices[i]];
      local_enriched_node_weights[i] = enriched_weights[local_dof_indices[i]];
    }
    else
    { 
      //if the node should be enriched
      if (cell->vertex(i).distance(wells[well_index]->center()) <= r_enr[well_index])
      {
        //if the node is from index subset N_w then the weight is 1 (else zero on blending element)
        local_enriched_node_weights[i] = 1;
        enriched_weights[local_dof_indices[i]] = 1;
      }
      
      //adding degree of freedom for every node on enriched element (both reproducing and blending elements)
      enriched_dof_indices[local_dof_indices[i]] = n_global_enriched_dofs;
      local_enriched_dofs[i] = n_global_enriched_dofs;
      n_global_enriched_dofs ++;
      
    }
    //std::cout << enriched_dof_indices[local_dof_indices[i]] << " ";
  }
  //std::cout << std::endl;
 
 
    /// Resolve polar quadrature:
    bool add_polar_quadrature = false;
    double temp_r = cell->diameter(),
           width = (well_band_width_ratio_+1) * wells[well_index]->radius();
    if( cell->center().distance(wells[well_index]->center()) < temp_r + width )
        add_polar_quadrature = use_polar_quadrature_;
 
  /////-----------------------------Well Boundary Integration part--------------start
  //if(well_computation_ == Well_computation::bc_newton)
  {
    //looking for quadrature poitns of the well in the current cell
    //temporary vector of q_points
    std::vector<const Point<2>* > points;
    //flag for addition cell to the vector in well object
    bool q_points_to_add = false;
    //checking if the q_points are in the cell
    for(unsigned int p=0; p < wells[well_index]->q_points().size(); p++)
    {
      if (cell->point_inside(wells[well_index]->q_points()[p]))
      {
        q_points_to_add = true;
        //adding point
        points.push_back( &(wells[well_index]->q_points()[p]) );
      }
    }
    
    //checking if there has been xdata already created
    //if not it is created
    //if so it means that XData object has been created and there are additions from other wells
    if(cell->user_pointer() == nullptr)
    {
      if(q_points_to_add)
      xdata_[m].push_back(new XDataCell(cell, 
                                    wells[well_index], 
                                    well_index, 
                                    local_enriched_dofs, 
                                    local_enriched_node_weights, 
                                    points));
      else
      xdata_[m].push_back(new XDataCell(cell, 
                                    wells[well_index], 
                                    well_index, 
                                    local_enriched_dofs, 
                                    local_enriched_node_weights));
    
      if(add_polar_quadrature)
          xdata_[m].back()->set_polar_quadrature(well_xquadratures_[well_index]);
      else
          xdata_[m].back()->set_polar_quadrature(nullptr);
          
      cell->set_user_pointer(xdata_[m].back());
    }
    else
    {
      XDataCell* xdata_pointer = static_cast<XDataCell*> (cell->user_pointer());
      if(q_points_to_add)
        xdata_pointer->add_data(wells[well_index], well_index, 
                                local_enriched_dofs, local_enriched_node_weights, points );
      else
        xdata_pointer->add_data(wells[well_index], well_index, 
                                local_enriched_dofs, local_enriched_node_weights );
        
      if(add_polar_quadrature &&  use_polar_quadrature_)
        xdata_pointer->set_polar_quadrature(well_xquadratures_[well_index]);
      else
        xdata_pointer->set_polar_quadrature(nullptr);
    }
  }
  /////-----------------------------Well Boundary Integration part--------------end
  /*
  //this will be possible if we know at this point whether the well crosses the cell (we know this only from q_points)
  else if(well_computation_ == Well_computation::sources)
  {
    //checking if there has been xdata already created
    //if not it is created
    //if so it means that XData object has been created and there are additions from other wells
    if(cell->user_pointer() == nullptr)
    {
      xdata.push_back(new XDataCell(cell, 
                                    wells[well_index], 
                                    well_index, 
                                    local_enriched_dofs, 
                                    local_enriched_node_weights));
    
      cell->set_user_pointer(xdata.back());
    }
    else
    {
      XDataCell* xdata_pointer = static_cast<XDataCell*> (cell->user_pointer());
        xdata_pointer->add_data(wells[well_index], well_index, 
                                local_enriched_dofs, local_enriched_node_weights );
    }
  }
  //*/
  
  // searching neighbors...
  for (unsigned int face_no=0; face_no < GeometryInfo<2>::faces_per_cell; ++face_no)
  {
    //std::cout << "\tface(" << face_no << "): ";
    typename DoFHandler<2>::face_iterator face = cell->face(face_no);
    
    //if the face is at the boundary, there is no neighbor, so continue
    if (face->at_boundary()) 
    {
      //std::cout << "at the boundary\n";
      continue;
    }
    
    // asking face about children - if so, then there must also be finer cells 
    // which are children or farther offsprings of our neighbor.
    if (face->has_children())
      {
        //std::cout << "\t Face has children." << std::endl;
        //TODO: check if adaptivity
        // iteration over subfaces - children
        for (unsigned int subface_no = 0; subface_no < face->number_of_children(); ++subface_no)
          {
            typename DoFHandler<2>::cell_iterator neighbor_child
                           = cell->neighbor_child_on_subface (face_no, subface_no);
            Assert (!neighbor_child->has_children(), ExcInternalError());
            
            //std::cout << "\tFace=" << face_no << " subface=" << subface_no 
            //          << " entering cell=" << neighbor_child->index() << std::endl;
            // entering on the neighbor's children behind the current face
            enrich_cell_blend(neighbor_child, well_index, enriched_dof_indices, 
                              enriched_weights, n_global_enriched_dofs,m);
          }
          //*/
      }
    else
      {
        // creating neighbor's cell iterator
        MASSERT(cell->neighbor(face_no).state() == IteratorState::valid,
            "Neighbor's state is invalid.");
        typename DoFHandler<2>::cell_iterator neighbor = cell->neighbor(face_no);
        MASSERT(!neighbor->has_children(), "Neighbor has children and is not active!");
        
        // asking if the neighbor is coarser, if not then it is neither coarser nor finer
        // so it is the same level of refinement
        if (!cell->neighbor_is_coarser(face_no) 
            //&&
            //(neighbor->index() > cell->index() ||
            //(neighbor->level() < cell->level() &&
            //neighbor->index() == cell->index() ))
           )
          {
            //std::cout << "\tneigh: " << neighbor->index() << "\t same refine level: " 
            //          << neighbor->level() << std::endl;
            enrich_cell_blend(neighbor, well_index, enriched_dof_indices, 
                              enriched_weights, n_global_enriched_dofs,m);
          } 
        else
          //is coarser
          {
            //std::cout << "\tneigh: " << neighbor->index() << "\t is coarser" << std::endl; 
            enrich_cell_blend(neighbor, well_index, enriched_dof_indices, 
                        enriched_weights, n_global_enriched_dofs,m);
          }
      }
  }
}

//-------------------------------------------------------------------------------------- ENRICH CELL
void XModel::enrich_cell ( const DoFHandler<2>::active_cell_iterator cell,
                           const unsigned int &well_index,
                           std::vector<unsigned int> &enriched_dof_indices,
                           unsigned int &n_global_enriched_dofs,
                           unsigned int m
                         )
{
  //std::cout << "CALL enrich_cell on: " << cell->index();
  // if the flag is set = we have been already there, so continue
    if ( cell->user_flag_set() == true)
    {
      //std::cout << "\thave already been at cell\n";
      return;
    }
  
  //sets user flag for the cell in which we have been
  cell->set_user_flag();
  
  //flag is true if no node is to be enriched (is not in the enrichment radius )
  bool cell_not_enriching = true;
  
  for(unsigned int i=0; i < fe.dofs_per_cell; i++)
  {
    if(cell->vertex(i).distance(wells[well_index]->center()) <= r_enr[well_index])
    {
      cell_not_enriching = false;
    }
  }
  
  //if there is no previously enriched dof we can return
  if (cell_not_enriching)
  {
    return;
  }
  //else
  
  
  std::vector<unsigned int> local_dof_indices(fe.dofs_per_cell);
  cell->get_dof_indices(local_dof_indices);
  
  //else we continue by enriching dofs
  //std::cout << "\tenriched: ";
  //temporary space for enriched dofs
  std::vector<unsigned int> local_enriched_dofs(fe.dofs_per_cell,0);
  std::vector<unsigned int> local_enriched_node_weights(fe.dofs_per_cell,0);    //not needed - is input to XDataCell::add(...)
  
  //enriching dofs
  /*
  const unsigned int weigth_boundary = -1;
    if(cell->at_boundary())
    {
        for (unsigned int face_no=0; face_no < GeometryInfo<2>::faces_per_cell; ++face_no)
        {
            //std::cout << "\tface(" << face_no << "): ";
            //typename DoFHandler<2>::face_iterator face = cell->face(face_no);
   
            //if the face is at the boundary, we do not enrich nodes
            if (cell->at_boundary(face_no)) 
            {
                //DBGMSG("dofs_per_cell: %d\n", fe.dofs_per_face);
                //DBGMSG("cell: %d, face: %d\n", cell->index(),face_no);
                for(unsigned int i=0; i < GeometryInfo<2>::vertices_per_face; i++)    
                    enriched_dof_indices[cell->face(face_no)->vertex_dof_index(i,0)] = weigth_boundary;
//                 for(unsigned int i=0; i < fe.dofs_per_face; i++)    
//                 enriched_weights[cell->face(face_no)->dof_index(i)] = weigth_boundary;
            }   
        }
    }
  //*/
  
  
  //enriching dofs

  for(unsigned int i=0; i < fe.dofs_per_cell; i++)
  {
    //if the dof is on the boundary
//     if(enriched_dof_indices[local_dof_indices[i]] == weigth_boundary) 
//     {
//         enriched_dof_indices[local_dof_indices[i]] = 0;
//         continue;
//     }
    
    //if the node was previously enriched
    if (enriched_dof_indices[local_dof_indices[i]] != 0)
    {
      //adding enriched dof that has been already defined from previous cell
      local_enriched_dofs[i] = enriched_dof_indices[local_dof_indices[i]];
    }
    else
    { 
      //if the node should be enriched
      if (cell->vertex(i).distance(wells[well_index]->center()) <= r_enr[well_index])
      {
        //adding degree of freedom for every (and only) ENRICHED node on enriched element
        enriched_dof_indices[local_dof_indices[i]] = n_global_enriched_dofs;
        local_enriched_dofs[i] = n_global_enriched_dofs;
        n_global_enriched_dofs ++;
      }
    }
    //std::cout << enriched_dof_indices[local_dof_indices[i]] << " ";
  }
  //std::cout << std::endl;
 
    
    /// Resolve polar quadrature:
    bool add_polar_quadrature = false;
    double temp_r = cell->diameter(),
           width = (well_band_width_ratio_+1) * wells[well_index]->radius();
    if( cell->center().distance(wells[well_index]->center()) < temp_r + width )
        add_polar_quadrature = use_polar_quadrature_;
        
    
  /////-----------------------------Well Boundary Integration part--------------start
  //if(well_computation_ == Well_computation::bc_newton)
  {
    //looking for quadrature poitns of the well in the current cell
    //temporary vector of q_points
    std::vector<const Point<2>* > points;
    //flag for addition cell to the vector in well object
    bool q_points_to_add = false;
    //checking if the q_points are in the cell
    for(unsigned int p=0; p < wells[well_index]->q_points().size(); p++)
    {
      if (cell->point_inside(wells[well_index]->q_points()[p]))
      {
        q_points_to_add = true;
        //adding point
        points.push_back( &(wells[well_index]->q_points()[p]) );
      }
    }
    
    //checking if there has been xdata already created
    //if not it is created
    //if so it means that XData object has been created and there are additions from other wells
    if(cell->user_pointer() == nullptr)
    {
      if(q_points_to_add)
      xdata_[m].push_back(new XDataCell(cell, 
                                    wells[well_index], 
                                    well_index, 
                                    local_enriched_dofs, 
                                    local_enriched_node_weights, 
                                    points));
      else
      xdata_[m].push_back(new XDataCell(cell, 
                                    wells[well_index], 
                                    well_index, 
                                    local_enriched_dofs, 
                                    local_enriched_node_weights));
    
      if(add_polar_quadrature)
          xdata_[m].back()->set_polar_quadrature(well_xquadratures_[well_index]);
      else
          xdata_[m].back()->set_polar_quadrature(nullptr);
      
      cell->set_user_pointer(xdata_[m].back());
    }
    else
    {
      XDataCell* xdata_pointer = static_cast<XDataCell*> (cell->user_pointer());
      if(q_points_to_add)
        xdata_pointer->add_data(wells[well_index], well_index, 
                                local_enriched_dofs, local_enriched_node_weights, points );
      else
        xdata_pointer->add_data(wells[well_index], well_index, 
                                local_enriched_dofs, local_enriched_node_weights );
        
      if(add_polar_quadrature && use_polar_quadrature_)
          xdata_pointer->set_polar_quadrature(well_xquadratures_[well_index]);
      else
          xdata_pointer->set_polar_quadrature(nullptr);
    }
  }
  /////-----------------------------Well Boundary Integration part--------------end
  /*
  //this will be possible if we know at this point whether the well crosses the cell (we know this only from q_points)
  else if(well_computation_ == Well_computation::sources)
  {
    //checking if there has been xdata already created
    //if not it is created
    //if so it means that XData object has been created and there are additions from other wells
    if(cell->user_pointer() == nullptr)
    {
      xdata.push_back(new XDataCell(cell, 
                                    wells[well_index], 
                                    well_index, 
                                    local_enriched_dofs, 
                                    local_enriched_node_weights));
    
      cell->set_user_pointer(xdata.back());
    }
    else
    {
      XDataCell* xdata_pointer = static_cast<XDataCell*> (cell->user_pointer());
        xdata_pointer->add_data(wells[well_index], well_index, 
                                local_enriched_dofs, local_enriched_node_weights );
    }
  }
  //*/
  
  // searching neighbors...
  for (unsigned int face_no=0; face_no < GeometryInfo<2>::faces_per_cell; ++face_no)
  {
    //std::cout << "\tface(" << face_no << "): ";
    typename DoFHandler<2>::face_iterator face = cell->face(face_no);
    
    //if the face is at the boundary, there is no neighbor, so continue
    if (face->at_boundary()) 
    {
      //std::cout << "at the boundary\n";
      continue;
    }
    
    // asking face about children - if so, then there must also be finer cells 
    // which are children or farther offsprings of our neighbor.
    if (face->has_children())
      {
        //std::cout << "\t Face has children." << std::endl;
        //TODO: check if adaptivity
        // iteration over subfaces - children
        for (unsigned int subface_no = 0; subface_no < face->number_of_children(); ++subface_no)
          {
            typename DoFHandler<2>::cell_iterator neighbor_child
                           = cell->neighbor_child_on_subface (face_no, subface_no);
            Assert (!neighbor_child->has_children(), ExcInternalError());
            
            //std::cout << "\tFace=" << face_no << " subface=" << subface_no 
            //          << " entering cell=" << neighbor_child->index() << std::endl;
            // entering on the neighbor's children behind the current face
            enrich_cell(neighbor_child, well_index, enriched_dof_indices, n_global_enriched_dofs,m);
          }
          //*/
      }
    else
      {
        // creating neighbor's cell iterator
        MASSERT(cell->neighbor(face_no).state() == IteratorState::valid,
            "Neighbor's state is invalid.");
        typename DoFHandler<2>::cell_iterator neighbor = cell->neighbor(face_no);
        MASSERT(!neighbor->has_children(), "Neighbor has children and is not active!");
        
        // asking if the neighbor is coarser, if not then it is neither coarser nor finer
        // so it is the same level of refinement
        if (!cell->neighbor_is_coarser(face_no) 
            //&&
            //(neighbor->index() > cell->index() ||
            //(neighbor->level() < cell->level() &&
            //neighbor->index() == cell->index() ))
           )
          {
            //std::cout << "\tneigh: " << neighbor->index() << "\t same refine level: " 
            //          << neighbor->level() << std::endl;
            enrich_cell(neighbor, well_index, enriched_dof_indices, n_global_enriched_dofs,m);
          } 
        else
          //is coarser
          {
            //std::cout << "\tneigh: " << neighbor->index() << "\t is coarser" << std::endl; 
            enrich_cell(neighbor, well_index, enriched_dof_indices, n_global_enriched_dofs,m);
          }
      }
  }
}


void XModel::setup_system()
{
    MASSERT(transmisivity_.size() == n_aquifers_, 
            "Wrong size of transmisivity vector (must be equal number of aquifers).");
    /// block and vector size initialization
    unsigned int n_blocks = n_aquifers_+1;
    block_matrix.resize(n_blocks);
    block_comm_matrix.resize(n_blocks);
    
    block_solution.reinit(n_blocks);
    block_system_rhs.reinit(n_blocks);
    
    system_matrix_.reinit(n_blocks, n_blocks);
  
    // clear xdata and node_enrich_values before new run
    for(unsigned int m=0; m < xdata_.size(); m++)
        for(unsigned int x=0; x < xdata_[m].size(); x++)
            delete xdata_[m][x];
  
    xdata_.clear();
    xdata_.resize(n_aquifers_);
    node_enrich_values.clear();
    node_enrich_values.resize(n_aquifers_);
 
    //prepare clean vector for cell pointers (pointer to xdata)
    tria_pointers_.clear();
    tria_pointers_.resize(n_aquifers_); 
    
    dof_handler->initialize(*triangulation,fe);
    
    if(hanging_nodes)
    {
        //HANGING NODES
        //clearing after the last refinement cycle
        hanging_node_constraints.clear();
        //making hanging nodes
        DoFTools::make_hanging_node_constraints (*dof_handler,
                                                hanging_node_constraints);
        //finalizing hanging nodes for this dof_handler
        hanging_node_constraints.close();
        //hanging_node_constraints.print(std::cout);
        //std::cout << "number of constrains: " << hanging_node_constraints.n_constraints() << std::endl;
    }
}


//------------------------------------------------------------------------------------------- SETUP    
void XModel::assemble_system ()
{
    // assembly on aquifers
    for(unsigned int m = 1; m <= n_aquifers_; m++)
    {
        std::cout << "######### aquifer" << m << " assembly ########## T = " << transmisivity_[m-1] << std::endl;
        setup_subsystem(m);
        assemble_subsystem(m);
        system_matrix_.enter(block_matrix[m],m,m);
    }
    
    assemble_communication();
//     DBGMSG("System matrix:\n");
//     system_matrix_.print_latex(cout);
    //DBGMSG("System RHS:\n");
    //block_system_rhs.print(cout);
    
    if(output_options_ & OutputOptions::output_sparsity_pattern)
    {
        //prints whole BlockSparsityPattern
        std::ofstream out1 (output_dir_+"block_sp_pattern.1");
        block_sp_pattern.print_gnuplot (out1);

//         //prints SparsityPattern of the block (0,0)
//         std::ofstream out2 (output_dir_+"00_sp_pattern.1");
//         block_sp_pattern.block(0,0).print_gnuplot (out2);
    }
}

void XModel::setup_subsystem(unsigned int m)
{
    //all data from previous run are cleared
    triangulation->clear_user_data();
 
    // before searching for enrichment cells, create polar quadratures
//     if( (well_xquadratures_.size() == 0)
//         && ( use_polar_quadrature_) )
    if(use_polar_quadrature_) compute_well_quadratures();
    
    //find cells which lies within the enrichment radius of the wells
    n_enriched_dofs_ = 0;
    find_enriched_cells(m-1);
    triangulation->save_user_pointers(tria_pointers_[m-1]);
  
    //adding well dofs to xdata
    for(unsigned int x=0; x < xdata_[m-1].size(); x++)
    {
        std::vector<unsigned int> well_dof_indices(xdata_[m-1][x]->n_wells(), 0);
        for(unsigned int w=0; w < xdata_[m-1][x]->n_wells(); w++)
        {
            //DBGMSG("setup-well_dof_indices: wi=%d \n", xdata[x]->get_well_index(w));
            well_dof_indices[w] = dof_handler->n_dofs() + n_enriched_dofs_ + xdata_[m-1][x]->get_well_index(w);
        }
        xdata_[m-1][x]->set_well_dof_indices(well_dof_indices);
    }
  
    //before using block_sp_pattern again, block_matrix must be cleared
    //to destroy pointer to block_sp_pattern
    //else the copy constructor will destroy block_sp_pattern
    //and block_matrix would point to nowhere!!
    if(! block_matrix[m].empty()) block_matrix[m].clear();
    //block_matrix[m] = 0.0;
    
    unsigned int n_fem = dof_handler->n_dofs(),
                 n_xfem = n_enriched_dofs_,
                 n_wells = wells.size(),
                 n_well_block = n_fem + n_xfem;
    unsigned int dimension = n_fem + n_xfem + n_wells;
    n_standard_dofs_ = n_fem;
    n_dofs_ = dimension;

    CompressedSparsityPattern block_c_sparsity(dimension, dimension);
    
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<unsigned int> local_dof_indices(dofs_per_cell);
  
    DoFHandler<2>::active_cell_iterator
        cell = dof_handler->begin_active(),
        endc = dof_handler->end();
    for (; cell!=endc; ++cell)
    {
        cell->get_dof_indices(local_dof_indices);
    
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
            block_c_sparsity.add(local_dof_indices[i],local_dof_indices[j]);    //block A
        
        if (cell->user_pointer() != nullptr)
        { 
            //A *a=static_cast<A*>(cell->user_pointer()); //from DEALII (TriaAccessor)
            XDataCell *cell_xdata = static_cast<XDataCell*>( cell->user_pointer() );
      
            for(unsigned int w=0; w < cell_xdata->n_wells(); w++)
            {
                for(unsigned int k=0; k < dofs_per_cell; k++)
                {
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                        if(cell_xdata->global_enriched_dofs(w)[k] != 0)
                        {
                //block R_transpose
                block_c_sparsity.add(local_dof_indices[i], cell_xdata->global_enriched_dofs(w)[k]);
                //block R
                block_c_sparsity.add(cell_xdata->global_enriched_dofs(w)[k], local_dof_indices[i]);
                //block C_transpose - enriched dofs
                block_c_sparsity.add(cell_xdata->global_enriched_dofs(w)[k], cell_xdata->get_well_dof_index(w));
                //block C - enriched dofs
                block_c_sparsity.add(cell_xdata->get_well_dof_index(w), cell_xdata->global_enriched_dofs(w)[k]);
                        }
              
                //block S
                        for(unsigned int ww=0; ww < cell_xdata->n_wells(); ww++)
                        {
                            if(cell_xdata->global_enriched_dofs(w)[i] != 0
                               && cell_xdata->global_enriched_dofs(ww)[k] != 0)
                block_c_sparsity.add(cell_xdata->global_enriched_dofs(w)[i], 
                                     cell_xdata->global_enriched_dofs(ww)[k]);
                        } // for ww
                    } // for i

                //block C, C_transpose - unenriched dofs
                    if(cell_xdata->q_points(w).size() > 0)        //does the well cross the cell?
                    {
                        block_c_sparsity.add(local_dof_indices[k], cell_xdata->get_well_dof_index(w));
                        block_c_sparsity.add(cell_xdata->get_well_dof_index(w), local_dof_indices[k]);
                    }
                } // for k
            } // for xdata->n_wells
        } // if user_pointer
    } // for cells
  
  
    XDataCell::initialize_node_values(node_enrich_values[m-1], xdata_[m-1], wells.size());
    DBGMSG("XData inicialization done - node values computed.\n");
    
    switch(enrichment_method_)
    {
        case Enrichment_method::xfem: 
            prepare_shape_well_averiges<Enrichment_method::xfem>(shape_well_averiges, 
                                                                 xdata_[m-1]);
            break;
        case Enrichment_method::xfem_ramp: 
            prepare_shape_well_averiges<Enrichment_method::xfem_ramp>(shape_well_averiges, 
                                                                      xdata_[m-1]);
        break;
        case Enrichment_method::xfem_shift:
            prepare_shape_well_averiges<Enrichment_method::xfem_shift>(shape_well_averiges, 
                                                                       xdata_[m-1]);
        break;
        case Enrichment_method::sgfem:
            prepare_shape_well_averiges<Enrichment_method::sgfem>(shape_well_averiges, 
                                                                  xdata_[m-1]);
        break;
    }
    
//     DBGMSG("Precomputed shape functions integral on the well edge:\n");
//     for(unsigned int w =0; w < wells.size(); w++) 
//     {
//         DBGMSG("Well %d:\n",w);
//         for(std::map<unsigned int,double>::iterator val = shape_well_averiges[w].begin(); val != shape_well_averiges[w].end(); ++val)
//         {
//             std::cout << "func number: " << val->first << " \t val: " << val->second << std::endl;
//         }
//     }
    
    //well averaging
    for (unsigned int w=0; w < wells.size(); ++w)
    {
        std::map<unsigned int,double>::iterator val_i = shape_well_averiges[w].begin();
        for(; val_i != shape_well_averiges[w].end(); ++val_i)
        {
            std::map<unsigned int,double>::iterator val_j = shape_well_averiges[w].begin();
            for(; val_j != shape_well_averiges[w].end(); ++val_j)
            {
                block_c_sparsity.add(val_i->first, val_j->first);
            }
        }
    }
    
    //diagonal pattern in the block (2,2)
    for(unsigned int w = 0; w < wells.size(); w++)
        block_c_sparsity.add(n_well_block + w,n_well_block + w);
  
    if(hanging_nodes)
    {
        //condensing hanging nodes
        hanging_node_constraints.condense(block_c_sparsity);
    }
  
    //copy from (temporary) BlockCompressedSparsityPattern to (main) BlockSparsityPattern
    block_sp_pattern.copy_from(block_c_sparsity);
 
    //reinitialization of block_matrix
    block_matrix[m].reinit(block_sp_pattern);
 
    //prints number of nozero elements in block_c_sparsity
    std::cout << "nozero elements in block_sp_pattern: " << block_sp_pattern.n_nonzero_elements() << std::endl;
  
    //reinitialization of block_solution 
    block_solution.block(m).reinit(dimension);
    block_system_rhs.block(m).reinit(dimension);
    
    //DBGMSG("Printing sparsity pattern: \n");
    //block_sp_pattern.print(std::cout);
    //std::cout << "\n\n";
}


void XModel::assemble_subsystem (unsigned int m)
{
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);        //HOMOGENOUS NEUMANN -> = 0, else source term
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  
    //for checking number of active cells (there used to be a problem in adaptivity)
    unsigned int count = 0;

    DoFHandler<2>::active_cell_iterator
        cell = dof_handler->begin_active(),
        endc = dof_handler->end();
    for (; cell!=endc; ++cell)
    {
        count++;
        fe_values.reinit (cell);
        cell_matrix = 0;
        cell_rhs = 0;		//HOMOGENOUS NEUMANN -> = 0
    
        /*
        //printing Jakobi determinants, fe_values flag must be set in constructors: update_jacobians
        //Jakobi determinant is constant for all quadrature points sofar
        DBGMSG("Jakobian unenriched cell(%d): ", cell->index());
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
        std::cout << "  " << fe_values.jacobian(q_point).determinant();
        }
        std::cout <<"\n";
        //*/
      
        if (cell->user_pointer() == nullptr)
        {
            cell->get_dof_indices (local_dof_indices);
        
            //INTEGRALS FOR BLOCK(0,0) ... matrix A
            for (unsigned int i=0; i < dofs_per_cell; ++i)
            for (unsigned int j=0; j < dofs_per_cell; ++j)
            for (unsigned int q_point=0; q_point < n_q_points; ++q_point)
                cell_matrix(i,j) += ( transmisivity_[m-1] *
                                    fe_values.shape_grad (i, q_point) *
                                    fe_values.shape_grad (j, q_point) *
                                    fe_values.JxW (q_point)
                                    );
            //FILLING MATRIX BLOCK A
            block_matrix[m].add(local_dof_indices, cell_matrix);
                
            if(rhs_function != nullptr)
            {
                // HOMOGENOUS NEUMANN -> = 0, else source term
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                    cell_rhs(i) += (fe_values.shape_value (i, q_point) *
                                rhs_function->value(fe_values.quadrature_point(q_point))
                                * fe_values.JxW (q_point));

                block_system_rhs.block(m).add(local_dof_indices, cell_rhs);
            }
        }    
        else
        {
            FullMatrix<double>   enrich_cell_matrix;
            std::vector<unsigned int> enrich_dof_indices; //dof indices of enriched and unrenriched dofs
            Vector<double>       enrich_cell_rhs; 
        
            //A *a=static_cast<A*>(cell->user_pointer()); //from DEALII (TriaAccessor)
            XDataCell * xdata = static_cast<XDataCell*>( cell->user_pointer() );
            
            if( (xdata->n_polar_quadratures() == 0) 
                || 
                ( ! use_polar_quadrature_) )
            {
                XQuadratureCell * xquadrature = new XQuadratureCell(xdata, 
                                                                    fe_values.get_mapping(), 
                                                                    XQuadratureCell::Refinement::edge);
                xquadrature->refine(adaptive_integration_refinement_level_);
//                 DBGMSG("cell %d - adaptive refinement level %d\n",cell->index(), xquadrature->level());
                
                if (output_options_ & OutputOptions::output_adaptive_plot)
                {
                    stringstream dir_name;
                    dir_name << "/adaptref_" << cycle_ << "/";
                    //output only cells which have well inside
                    //if(t == adaptive_integration_refinement_level_-1)
            //         (output_dir, false, true) must be set to unit coordinates and to show on screen 
                    xquadrature->gnuplot_refinement(create_subdirectory(output_dir_,dir_name.str()));
                }
                
                Adaptive_integration adaptive_integration(xdata,fe,(XQuadratureBase *)xquadrature,m);
                
                //sets the dirichlet and source function
                if(dirichlet_function || rhs_function)
                    adaptive_integration.set_functors(dirichlet_function, rhs_function);
                
                switch(enrichment_method_)
                {
                    case Enrichment_method::xfem: 
                        adaptive_integration.integrate<Enrichment_method::xfem>(enrich_cell_matrix, 
                                                                                enrich_cell_rhs, 
                                                                                enrich_dof_indices, 
                                                                                transmisivity_[m-1]);
                        break;
                    case Enrichment_method::xfem_ramp: 
                        //adaptive_integration.integrate_xfem(enrich_cell_matrix, enrich_cell_rhs, enrich_dof_indices, transmisivity[0]);
                        adaptive_integration.integrate<Enrichment_method::xfem_ramp>(enrich_cell_matrix, 
                                                                                    enrich_cell_rhs, 
                                                                                    enrich_dof_indices, 
                                                                                    transmisivity_[m-1]);
                    break;
                    case Enrichment_method::xfem_shift:
                    adaptive_integration.integrate<Enrichment_method::xfem_shift>(enrich_cell_matrix, 
                                                                                enrich_cell_rhs, 
                                                                                enrich_dof_indices, 
                                                                                transmisivity_[m-1]);
                    break;
                    case Enrichment_method::sgfem:
                    adaptive_integration.integrate<Enrichment_method::sgfem>(enrich_cell_matrix, 
                                                                            enrich_cell_rhs, 
                                                                            enrich_dof_indices, 
                                                                            transmisivity_[m-1]);
                    break;
                }
            
            } // if
            else
            {
                XQuadratureCell * xquadrature = new XQuadratureCell(xdata, 
                                                                    fe_values.get_mapping(), 
                                                                    XQuadratureCell::Refinement::polar);
                xquadrature->refine(adaptive_integration_refinement_level_);
//                 DBGMSG("cell %d - polar adaptive refinement level %d\n",cell->index(), xquadrature->level());
                
                if (output_options_ & OutputOptions::output_adaptive_plot)
                {
                    stringstream dir_name;
                    dir_name << "/adaptref_" << cycle_ << "/";
                        
                    //output only cells which have well inside
                    //if(t == adaptive_integration_refinement_level_-1)
            //         (output_dir, false, true) must be set to unit coordinates and to show on screen 
                    xquadrature->gnuplot_refinement(create_subdirectory(output_dir_, dir_name.str()));
                }
                    
                AdaptiveIntegrationPolar adaptive_integration_polar(xdata,fe,
                                                                    (XQuadratureBase *)xquadrature,
                                                                    xdata->polar_quadratures(),
                                                                    m);
                
                //sets the dirichlet and source function
                if(dirichlet_function || rhs_function)
                    adaptive_integration_polar.set_functors(dirichlet_function, rhs_function);
                
                switch(enrichment_method_)
                {
                    case Enrichment_method::xfem: 
                        adaptive_integration_polar.integrate<Enrichment_method::xfem>(enrich_cell_matrix, 
                                                                                enrich_cell_rhs, 
                                                                                enrich_dof_indices, 
                                                                                transmisivity_[m-1]);
                        break;
                    case Enrichment_method::xfem_ramp: 
                        //adaptive_integration.integrate_xfem(enrich_cell_matrix, enrich_cell_rhs, enrich_dof_indices, transmisivity[0]);
                        adaptive_integration_polar.integrate<Enrichment_method::xfem_ramp>(enrich_cell_matrix, 
                                                                                    enrich_cell_rhs, 
                                                                                    enrich_dof_indices, 
                                                                                    transmisivity_[m-1]);
                    break;
                    case Enrichment_method::xfem_shift:
                    adaptive_integration_polar.integrate<Enrichment_method::xfem_shift>(enrich_cell_matrix, 
                                                                                enrich_cell_rhs, 
                                                                                enrich_dof_indices, 
                                                                                transmisivity_[m-1]);
                    break;
                    case Enrichment_method::sgfem:
                    adaptive_integration_polar.integrate<Enrichment_method::sgfem>(enrich_cell_matrix, 
                                                                            enrich_cell_rhs, 
                                                                            enrich_dof_indices, 
                                                                            transmisivity_[m-1]);
                    break;
                }
            }
            //printing enriched nodes and dofs
//               DBGMSG("Printing dof_indices:  [");
//               for(unsigned int a=0; a < enrich_dof_indices.size(); a++)
//               {
//                   std::cout << std::setw(3) << enrich_dof_indices[a] << "  ";
//               }
//               std::cout << "]" << std::endl;
            
                //FILLING MATRIX BLOCKs
            block_matrix[m].add(enrich_dof_indices,enrich_cell_matrix);
            block_system_rhs.block(m).add(enrich_dof_indices,enrich_cell_rhs);
        } //else
    } //end for(cells)
  
    assemble_well_permeability_term(m);
    
    unsigned int w_idx = block_matrix[m].n() - wells.size();
    for (unsigned int w = 0; w < wells.size(); w++)
    {
        //addition to block (2,2) ... matrix E
        double temp_val = wells[w]->perm2aquitard(m) + wells[w]->perm2aquitard(m-1);
        block_matrix[m].add(w_idx,w_idx,temp_val);
        w_idx++;
    }
    
    DBGMSG("N_active_cells checkout(on triangulation,integrated): %d \t %d\n", triangulation->n_active_cells() ,count);
  
//   DBGMSG("Printing block matrix:\n");
//   //block_matrix[m].block(0,0).print_formatted(std::cout);
//   std::cout << "\n\n";
//   block_matrix[m].print_formatted(std::cout);
//   std::cout << "\n\n";
//   block_matrix[m].block(2,2).print_formatted(std::cout);
    //if(m == 0)
//     block_matrix[m].print(cout);
    assemble_dirichlet(m);
//     DBGMSG("block_matrix[%d]:\n",m);
//     block_matrix[m].print(cout);
  
    if(hanging_nodes)
    {
        hanging_node_constraints.condense(block_matrix[m]);
        hanging_node_constraints.condense(block_system_rhs.block(m));
    }
    
    if( use_polar_quadrature_)
    {
        DBGMSG("N polar quadrature points check: %d %d\n", well_xquadratures_[0]->size(), AdaptiveIntegrationPolar::n_point_check);
        AdaptiveIntegrationPolar::n_point_check = 0;
        
        std::cout << "Total number of quadrature points used on enriched cells (cell + polar): " 
                  << Adaptive_integration::n_enrich_quad_points << " + "
                  << AdaptiveIntegrationPolar::n_enrich_quad_points << " = "
                  << AdaptiveIntegrationPolar::n_enrich_quad_points + Adaptive_integration::n_enrich_quad_points 
                  <<std::endl;
        AdaptiveIntegrationPolar::n_enrich_quad_points = 0;
        Adaptive_integration::n_enrich_quad_points = 0;
    }
    else{
        std::cout << "Total number of quadrature points used on enriched cells: " 
                  << Adaptive_integration::n_enrich_quad_points <<std::endl;
        Adaptive_integration::n_enrich_quad_points = 0;
    }
    
    
}

void XModel::assemble_well_permeability_term(unsigned int m)
{
    for (unsigned int w=0; w < wells.size(); ++w)
    {
        Well * well = wells[w];
        
        std::map<unsigned int,double>::iterator val_i = shape_well_averiges[w].begin();
        for(; val_i != shape_well_averiges[w].end(); ++val_i)
        {
            std::map<unsigned int,double>::iterator val_j = shape_well_averiges[w].begin();
            for(; val_j != shape_well_averiges[w].end(); ++val_j)
            {
                //if(block_sp_pattern.exists(val_i->first,val_j->first))
                {
                double value = well->perm2aquifer(m-1) / well->circumference()
                               * val_i->second 
                               * val_j->second;
                block_matrix[m].add(val_i->first,
                                    val_j->first,
                                    value);
                }
            }
        }          
    }
}


// void XModel::assemble_reduce_known(unsigned int m)
// {
//     std::map<unsigned int,double> known_values;
//     
// //     for(XDataCell *loc_xdata: xdata)
// //     {
// //         if(loc_xdata->q_points().size() > 0)
// //         {
// //             
// //         }
// //     }
//     unsigned int offset = dof_handler->n_dofs() + n_enriched_dofs;
//     for(unsigned int w=0; w < wells.size(); w++)
//     {
//         known_values[offset+w] = wells[w]->pressure();
//     }
//     
//     //TODO: cannot do (block_matrix, vector, vector)
// //     MatrixTools::apply_boundary_values(known_values,
// //                                        block_matrix[m],
// //                                        block_solution[m], 
// //                                        block_system_rhs[m],
// //                                        true
// //                                       );
// }

void XModel::assemble_communication()
{
    unsigned int size = block_matrix[1].n(),
                 offset = size - wells.size();
                
    CompressedSparsityPattern c_sparsity0(size, size),                  //communication between aquifers
                              c_sparsity1(wells.size(), size),          //communication on the top
                              c_sparsity2(wells.size(), wells.size());  //first diagonal well block
    for (unsigned int j=0; j<wells.size(); j++)
    {
        c_sparsity0.add(j+offset, j+offset);
        c_sparsity1.add(j, j+offset);
        c_sparsity2.add(j, j);
    }

    comm_sp_pattern.resize(3);
    comm_sp_pattern[0].copy_from(c_sparsity0);
    comm_sp_pattern[1].copy_from(c_sparsity1);
    comm_sp_pattern[2].copy_from(c_sparsity2);
    
    for(unsigned int m = 1; m < n_aquifers_; m++)
    {
        block_comm_matrix[m].reinit(comm_sp_pattern[0]);
        
        for (unsigned int w=0; w < wells.size(); w++)
            block_comm_matrix[m].set(w+offset, w+offset, -wells[w]->perm2aquitard(m));
        
        //DBGMSG("block_comm_matrix[%d]:\n",m);
        //block_comm_matrix[m].print(cout);
        
        system_matrix_.enter(block_comm_matrix[m], m+1, m);
        system_matrix_.enter(block_comm_matrix[m], m, m+1);
    }
    
    //communication on the top
    block_comm_matrix[0].reinit(comm_sp_pattern[1]);
    block_matrix[0].reinit(comm_sp_pattern[2]);
    block_solution.block(0).reinit(wells.size());
    block_system_rhs.block(0).reinit(wells.size());
    unsigned int w_idx = size - wells.size();
    for (unsigned int w=0; w < wells.size(); w++)
    {
        double perm2aquitard = wells[w]->perm2aquitard(0),
               mat_diag = perm2aquitard;                                            // c^{M+1}_w
                          //+ 2*M_PI*wells[w]->radius()*wells[w]->perm2aquifer(0),   // 2piR_w*sigma_w
               //elimination_coef = - perm2aquitard / mat_diag;
        if(wells[w]->is_pressure_set()) 
        {        
            //DBGMSG("Dirichlet well pressure, %d.\n", w_idx);
            //dirichlet boundary elimination
//             block_matrix[1].set(w_idx, w_idx, 
//                                 block_matrix[1](w_idx,w_idx) + elimination_coef * perm2aquitard);
//             block_comm_matrix[1].set(w_idx, w_idx, 
//                                 block_comm_matrix[1](w_idx,w_idx) + elimination_coef * perm2aquitard);
            block_system_rhs.block(1)(w_idx) = block_system_rhs.block(1)(w_idx) 
                                               // elimination_coef * mat_diag = -1
                                               + perm2aquitard * wells[w]->pressure();
            block_matrix[0].set(w,w,1.0);
            block_solution.block(0)(w) = wells[w]->pressure();
            block_system_rhs.block(0)(w) = wells[w]->pressure();
        }
        else
        {
            block_matrix[0].set(w,w, mat_diag);
            block_comm_matrix[0].set(w, w+offset, -perm2aquitard);
        }
        w_idx++;
    }   
//     DBGMSG("block_comm_matrix[%d]:\n",0);
//     block_comm_matrix[0].print(cout);    
//     block_matrix[0].print(cout);    
//     block_matrix[1].print(cout);
    system_matrix_.enter(block_matrix[0], 0, 0);
    system_matrix_.enter(block_comm_matrix[0], 0, 1);
    system_matrix_.enter(block_comm_matrix[0], 1, 0, 1.0, true);
    
}

void XModel::assemble_dirichlet(unsigned int m)
{
   // MASSERT(dirichlet_function != NULL, "Dirichlet BC function has not been set.\n");
    MASSERT(dof_handler != NULL, "DoF Handler object does not exist.\n");

    std::map<unsigned int,double> boundary_values;
    if(m == n_aquifers_)
            VectorTools::interpolate_boundary_values (*dof_handler,
                                            0,
                                            ZeroFunction<2>(),
                                            boundary_values);
//     else if(m == 1)
//             VectorTools::interpolate_boundary_values (*dof_handler,
//                                             0,
//                                             *dirichlet_function,
//                                             boundary_values);
    else return;
    
   DBGMSG("boundary_values size = %d\n",boundary_values.size());
   MatrixTools::apply_boundary_values (boundary_values,
                                       block_matrix[m],
                                       block_solution.block(m),
                                       block_system_rhs.block(m),
                                       true
                                      );
   
   std::cout << "Dirichlet BC assembled succesfully." << std::endl;
}

                               
    
void XModel::solve ()
{
  //how to do things for BLOCK objects
  //http://www.dealii.org/archive/dealii/msg02097.html
  
    SolverControl	solver_control(solver_max_iter_, solver_tolerance_);
    GrowingVectorMemory<BlockVector<double> > vector_memory;
 
  
  /*
  //EIGENVALUES ESTIMATE
  
  double biggest_eigen = 0, 
         smallest_eigen = 0;
  double range = 1.0;
  BlockVector<double>::iterator iter = block_solution.begin(),
                                end = block_solution.end();
  for(iter; iter < end; ++iter)
  {
    *iter = (rand() / (double)RAND_MAX) * range;
  }
  EigenPower<BlockVector<double> > eigen_power(solver_control,vector_memory);
  eigen_power.solve<BlockSparseMatrix<double> >(biggest_eigen, block_matrix, block_solution);
  std::cout << std::scientific << "Solver: steps: " << solver_control.last_step() << "\t residuum: " << setprecision(16) << solver_control.last_value() << std::endl;
  std::cout << "Biggest EigenValue = " << biggest_eigen << endl;
  
  
  for(iter = block_solution.begin(); iter < end; ++iter)
  {
    *iter = (rand() / (double)RAND_MAX) * range;
  }
  EigenPower<BlockVector<double> > eigen_power2(solver_control,vector_memory,biggest_eigen);
  eigen_power.solve<BlockSparseMatrix<double> >(smallest_eigen, block_matrix, block_solution);
  std::cout << std::scientific << "Solver: steps: " << solver_control.last_step() << "\t residuum: " << setprecision(16) << solver_control.last_value() << std::endl;
  std::cout << "Smallest EigenValue = " << smallest_eigen << endl;
  
  std::cout << "Estimate of condition number = " << biggest_eigen / smallest_eigen << endl;
  
  for(iter = block_solution.begin(); iter < end; ++iter)
  {
    *iter = 0;
  }
  //*/
  
  
  /*
  //SOLVER SELECTOR
  SolverSelector<BlockVector<double> > solver_selector;
  solver_selector.set_control(solver_control);
  solver_selector.select("cg");
  //*/
  
  
  //USING CG, BICG, PreconditionJacobi
  //SolverBicgstab<BlockVector<double> > solver_bicg(solver_control,vector_memory);
  SolverCG<BlockVector<double> > solver_cg(solver_control, vector_memory, 
                                           SolverCG<BlockVector<double> >::AdditionalData(false, false,//true, 
                                                                                          false, false));
  
    // block Jacobi preconditioning
    BlockTrianglePrecondition<double> preconditioning(n_aquifers_+1);
    std::vector<SparseMatrix<double> > precond_mat(n_aquifers_+1);
    
    precond_mat[0].reinit(comm_sp_pattern[2]);
    for (unsigned int j=0; j<wells.size(); ++j)
        precond_mat[0].add(j,j, 1.0/block_matrix[0](j,j));
    preconditioning.enter(precond_mat[0], 0, 0);
    
    unsigned int size = block_matrix[1].n();
    SparsityPattern sp_pattern;
    CompressedSparsityPattern c_sparsity(size, size);
    for (unsigned int j=0; j<size; ++j)
        c_sparsity.add(j,j);
    
    sp_pattern.copy_from(c_sparsity);
    
    for(unsigned int m = 1; m <= n_aquifers_; m++)
    {
        precond_mat[m].reinit(sp_pattern);
        for (unsigned int j=0; j<size; ++j)
        {
            //DBGMSG("m=%d, j=%d\n",m,j);
            precond_mat[m].add(j,j, 1.0/block_matrix[m](j,j));
        }
        preconditioning.enter(precond_mat[m], m, m);
    }

    DBGMSG("Dimension check:\n");
    for(unsigned int m = 0; m <= n_aquifers_; m++)
    {
        unsigned int w = 3;
        std::cout << "matrix[" << m << "]  " << std::setw(w) << block_matrix[m].m() << " x " << std::setw(w) << block_matrix[m].n()
                  << "\t precond_matrix[" << m << "]  " << std::setw(w) << precond_mat[m].m() << " x " << std::setw(w) << precond_mat[m].n(); 
        if(m != n_aquifers_) 
            std::cout << "\t comm_matrix[" << m << "]  " << std::setw(w) << block_comm_matrix[m].m() << " x " << std::setw(w) << block_comm_matrix[m].n();
        
        std::cout <<"\t RHS[" << m << "] " << std::setw(w) << block_system_rhs.block(m).size() << 
                    "\t SOL[" << m << "] " << std::setw(w) << block_solution.block(m).size() << std::endl;
    }
    
  solver_cg.solve(system_matrix_, block_solution, block_system_rhs, preconditioning); //PreconditionIdentity());
  
  solver_iterations_ = solver_control.last_step();
  
  std::cout << std::scientific << "Solver: steps: " << solver_control.last_step() << "\t residuum: " 
            << setprecision(4) << solver_control.last_value() << std::endl;
  //*/
 
    preconditioning.clear();
  
  
  /*
  // DIRECT SOLVER
  SparseDirectUMFPACK direct_solver;
  direct_solver.initialize(block_matrix);
  direct_solver.solve(block_system_rhs);
  block_solution = block_system_rhs;
  //*/
  
  //TODO: do it after blocks
  if(hanging_nodes)
  {
    //hanging_node_constraints.distribute(block_solution);
  }
  
//   for (unsigned int w=0; w < wells.size(); ++w)
//       std::cout << setprecision(12) << "value of H" << w << " = " << block_solution[0].block(2)[w] << std::endl;
  
//   DBGMSG("Printing solution:\n");
//   block_solution.print(std::cout);
  precond_mat.clear();
}


void XModel::output_results (const unsigned int cycle)
{ 
    // MATRIX OUTPUT
    if(output_options_ & OutputOptions::output_matrix)
    {
        //TODO: output whole system matrix
        std::stringstream matrix_name;
        matrix_name << "matrix_" << cycle;
        //write_block_sparse_matrix(block_matrix[0],matrix_name.str());
    }
  
    
    // MESH OUTPUT
    std::stringstream filename; 
    filename << output_dir_ << "xfem_mesh_" << cycle;
    
    if(output_options_ & OutputOptions::output_gmsh_mesh)
    {
        std::ofstream output (filename.str() + ".msh");
        GridOut grid_out;
        grid_out.write_msh<2> (*triangulation, output);
        std::cout << "\nXFEM  gmsh mesh written in:\t" << filename.str() << ".msh" << std::endl;
    }
   
    // dummy solution for displaying mesh in Paraview
    if(output_options_ & OutputOptions::output_vtk_mesh)
    {
        DataOut<2> data_out;
        data_out.attach_dof_handler (*dof_handler);
        Vector<double> dummys_solution(dof_handler->n_dofs());
        data_out.add_data_vector (dummys_solution, "xfem_grid");
        data_out.build_patches (0);
        std::ofstream output (filename.str()+".vtk");
        data_out.write_vtu (output); 
        std::cout << "\nXFEM vtk mesh written in:\t" << filename.str() << ".vtk" << std::endl;
    }
  
  /*
  //computing solution on the computational mesh
  std::vector< Point< 2 > > support_points(dof_handler->n_dofs());
  DoFTools::map_dofs_to_support_points<2>(fe_values.get_mapping(), *dof_handler, support_points);
  
  data_out.clear();
  data_out.attach_dof_handler (*dof_handler);
  
  std::cout << "computing solution on computational mesh" << std::endl;
  
  compute_distributed_solution(support_points);
  
  if(out_decomposed)
  {
    data_out.add_data_vector (dist_unenriched, "xfem_unenriched");
    data_out.add_data_vector (dist_enriched, "xfem_enriched"); 
  }
  data_out.add_data_vector (dist_solution, "xfem_solution");
  
  data_out.build_patches ();

  std::stringstream filename;
  filename << output_dir_ << "xmodel_solution_" << cycle << ".vtk";
   
  std::ofstream output (filename.str());
  data_out.write_vtk (output);
  //*/
  
  
  
    //WRITE THE DATA ON ADAPTIVELY REFINED MESH
    if(output_options_ & OutputOptions::output_solution)
    {
        DBGMSG("tria: %d  %d \n",triangulation->n_refinement_steps(), triangulation->n_levels());
        for(unsigned int m = 1; m <= n_aquifers_; m++)
        {
            if(output_triangulation) delete output_triangulation;
            
            output_triangulation = new PersistentTriangulation<2>(coarse_tria);
            PersistentTriangulation<2> &output_grid = *output_triangulation;
            
            triangulation->clear_user_data();
            triangulation->load_user_pointers(tria_pointers_[m-1]); //reload proper xdata
            output_grid.copy_triangulation(*triangulation);
            output_grid.restore();
            output_grid.clear_user_flags();
            
            FE_Q<2> temp_fe(1);
            DoFHandler<2> temp_dof_handler;
            temp_dof_handler.initialize(output_grid,temp_fe);
            
            Vector<double>::iterator first = block_solution.block(m).begin();
            Vector<double>::iterator last = first + dof_handler->n_dofs();
            dist_unenriched = Vector<double>(first, last);
            dist_solution = dist_unenriched;
            
            double tolerance = output_element_tolerance_;
            unsigned int iterations = 30;
            
            //TODO: template function
            switch(enrichment_method_)
            {
                case Enrichment_method::xfem: 
                //MASSERT(0,"Not implemented yet.");
                for(unsigned int n = 0; n < iterations; n++)
                {
                    DBGMSG("aquifer [%d] - output [%d]:\n", m, n);
                    if( recursive_output<Enrichment_method::xfem>(
                                    tolerance, output_grid, temp_dof_handler, temp_fe, n, m) )
                    break;
                }
                break;
                case Enrichment_method::xfem_ramp: 
                //MASSERT(0,"Not implemented yet.");
                for(unsigned int n = 0; n < iterations; n++)
                {
                    DBGMSG("aquifer [%d] - output [%d]:\n", m, n);
                    if( recursive_output<Enrichment_method::xfem_ramp>(
                                    tolerance, output_grid, temp_dof_handler, temp_fe, n, m) )
                    break;
                }
                break;
                
                case Enrichment_method::xfem_shift:
                for(unsigned int n = 0; n < iterations; n++)
                {
                    DBGMSG("aquifer [%d] - output [%d]:\n", m, n);
                    if( recursive_output<Enrichment_method::xfem_shift>(
                                    tolerance, output_grid, temp_dof_handler, temp_fe, n, m) )
                    break;
                }
                break;
                
                case Enrichment_method::sgfem:
                for(unsigned int n = 0; n < iterations; n++)
                {
                    DBGMSG("aquifer [%d] - output [%d]:\n", m, n);
                    if( recursive_output<Enrichment_method::sgfem>(
                                    tolerance, output_grid, temp_dof_handler, temp_fe, n, m) )
                    break;
                }
            } // switch
        } //for m
    } //if output_solution
}


void XModel::find_dofs_enriched_cells(std::vector<DoFHandler<2>::active_cell_iterator> &cells, 
                                      const unsigned int &dof_index,
                                      unsigned int m
                                     )
{
  cells.clear();
  for(unsigned int i=0; i < xdata_[m].size(); i++)
  {
    for(unsigned int w=0; w < xdata_[m][i]->n_wells(); w++)
      for(unsigned int k=0; k < xdata_[m][i]->global_enriched_dofs(w).size(); k++)
      {
        if(xdata_[m][i]->global_enriched_dofs(w)[k] == dof_index)
          cells.push_back(xdata_[m][i]->get_cell());
      }
  }
}

void XModel::get_dof_func(const std::vector< Point< 2 > >& points, 
                          const unsigned int& dof_index,
                          dealii::Vector< double >& dof_func, 
                          bool xfem)
{
  MASSERT(points.size() == dof_func.size(), "Vector of solution and support points must be of the same size.");
  if(xfem)
  {
    MASSERT(dof_handler->n_dofs() <= dof_index && dof_index < n_enriched_dofs_+dof_handler->n_dofs(), 
            "xfem is true. Given dof index is not index of enriched dof.");
  }
  else
  {
    MASSERT(dof_index < dof_handler->n_dofs(), 
            "xfem is false. Given dof index is not index of unenriched dof.");
  }
    
  std::vector<DoFHandler<2>::active_cell_iterator> cells(GeometryInfo<2>::faces_per_cell);  
  // finds all cells (maximum 4) that has given dof at its node
  find_dofs_enriched_cells(cells, dof_index, 0);
  
  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  //std::pair<DoFHandler<2>::active_cell_iterator, Point<2> > cell_and_point;
  XDataCell *cell_xdata;
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  
  Point<2> unit_point;
  
  for (unsigned int p = 0; p < points.size(); p++)
  {
    dof_func[p] = 0;
    
    //unsigned int c = 0;
    for(unsigned int c = 0; c < cells.size(); c++)
    {
      if(cells[c]->point_inside(points[p]))
      { 
        fe_values.reinit (cells[c]);
        unit_point = fe_values.get_mapping().transform_real_to_unit_cell(cells[c],points[p]);
        //cells[c]->get_dof_indices(local_dof_indices);
        
        if(xfem)
        {
          MASSERT(cells[c]->user_pointer() != nullptr,"Cell not enriched!");
        
          cell_xdata = static_cast<XDataCell*>( cells[c]->user_pointer() );
      
          for(unsigned int w = 0; w < cell_xdata->n_wells(); w++)
          {
            for(unsigned int k = 0; k < cell_xdata->global_enriched_dofs(w).size(); k++)
            {
              if(cell_xdata->global_enriched_dofs(w)[k] == dof_index)        
              {
                //for(unsigned int l = 0; l < dofs_per_cell; l++)
                {
                  //if(cell_xdata->global_enriched_dofs()[w][l] !=0 ) //blending function
                  {
                  dof_func[p] += // block_solution(dof_index) *
                                 //fe.shape_value(l, unit_point) * 
                                 fe.shape_value(k, unit_point) *
                                 (cell_xdata->get_well(w)->global_enrich_value(points[p]) 
                                  - cell_xdata->get_well(w)->global_enrich_value(cells[c]->vertex(k))
                                );
                  }
                } //for l
                break;
                //w = cell_xdata->n_wells();
                //k = cell_xdata->global_enriched_dofs(w).size();
              } //if
            } //for k
          } //for w
        } //if(xfem)
        else 
        {
          for(unsigned int i=0; i < dofs_per_cell; i++)
          {
            dof_func[p] += // block_solution(dof_index) *
                           fe.shape_value(i, unit_point);
          }
        }
        
        break;  //preventing sum of values of different cells on faces
      }
    }

  } //for p
  
  /*
  //iteration over all points where we compute solution
  for (unsigned int p = 0; p < points.size(); p++)
  {
    dof_func[p] = 0;
    //DBGMSG("point number: %d\n", p);
    //finds cell where points[p] lies and maps that point to unit_point
    //returns pair<cell, unit_point>
    // cell = cell_and_point.first
    // point = cell_and_point.second
    cell_and_point = GridTools::find_active_cell_around_point<2>(fe_values.get_mapping(), dof_handler, points[p]);
    
    unit_point = GeometryInfo<2>::project_to_unit_cell(cell_and_point.second); //due to roundoffs
    
    fe_values.reinit (cell_and_point.first);
    cell_and_point.first->get_dof_indices(local_dof_indices);

    if(!xfem)
    {
      for(unsigned int j=0; j < dofs_per_cell; j++)
      {
        if(dof_index == local_dof_indices[j])
        dof_func[p] += block_solution(local_dof_indices[j]) *
                      fe.shape_value(j, unit_point);
      }
    }

    
    if (cell_and_point.first->user_pointer() != nullptr && xfem)
    {
      cell_xdata = static_cast<XData*>( cell_and_point.first->user_pointer() );
      
      for(unsigned int w = 0; w < cell_xdata->wells().size(); w++)
      {
        for(unsigned int k = 0; k < cell_xdata->global_enriched_dofs()[w].size(); k++)
        {
          if(cell_xdata->global_enriched_dofs()[w][k] == dof_index)        
          {
            for(unsigned int l = 0; l < dofs_per_cell; l++)
            {
              dof_func[p] += block_solution(cell_xdata->global_enriched_dofs()[w][k]) *
                             fe.shape_value(l, unit_point) * 
                             fe.shape_value(k, unit_point) *
                             cell_xdata->wells()[w]->global_enrich_value(points[p]);
            } //for l
          } //if
        } //for k
      } //for w
    } //if
  } //for p
  //*/
}


void XModel::output_distributed_solution(const dealii::Triangulation< 2 > &dist_tria, const unsigned int& cycle, const unsigned int& m_aquifer)
{
    DataOut<2> data_out;
  
    QGauss<2>        dist_quadrature(2);
    FE_Q<2>          dist_fe(1);                    
    DoFHandler<2>    dist_dof_handler;
    FEValues<2>      dist_fe_values(dist_fe, dist_quadrature, update_default);
    ConstraintMatrix dist_hanging_node_constraints;

    //====================distributing dofs
    dist_dof_handler.initialize(dist_tria,dist_fe);
    
    DoFTools::make_hanging_node_constraints (dist_dof_handler, dist_hanging_node_constraints);  
    dist_hanging_node_constraints.close();

    //====================computing solution on the triangulation
    std::vector< Point< 2 > > support_points(dist_dof_handler.n_dofs());
    
    DoFTools::map_dofs_to_support_points<2>(dist_fe_values.get_mapping(), dist_dof_handler, support_points);
    std::cout << "Distributing solution on a given mesh..." << std::endl;
    std::cout << "...number of nodes in the mesh:   " << dist_dof_handler.n_dofs() << std::endl;
    std::cout << "...number of nodes in the xfem mesh:   " << dof_handler->n_dofs() << std::endl;
    std::cout << "...number of dofs in the xfem mesh:   " << dof_handler->n_dofs() << " unenriched and " 
                << n_enriched_dofs_ << " enriched" << std::endl;

    //====================vtk output
    //DataOut<2> data_out;
    data_out.attach_dof_handler (dist_dof_handler);
    
    std::cout << "computing solution on given mesh" << std::endl;
            
    switch(enrichment_method_)
    {
        case Enrichment_method::xfem: 
            compute_distributed_solution<Enrichment_method::xfem>(support_points);
            break;
        case Enrichment_method::xfem_ramp: 
            compute_distributed_solution<Enrichment_method::xfem_ramp>(support_points);
            break;
        
        case Enrichment_method::xfem_shift:  
            compute_distributed_solution<Enrichment_method::xfem_shift>(support_points);
            break;
        
        case Enrichment_method::sgfem:
            compute_distributed_solution<Enrichment_method::sgfem>(support_points);
            break;
        default: 
            MASSERT(0,"Unknown enrichment type or not implemented.");
    }
    
    dist_hanging_node_constraints.distribute(dist_unenriched);
    dist_hanging_node_constraints.distribute(dist_enriched);
    dist_hanging_node_constraints.distribute(dist_solution);
    
    if(output_options_ & OutputOptions::output_decomposed)
    {
        data_out.add_data_vector (dist_unenriched, "xfem_unenriched");
        data_out.add_data_vector (dist_enriched, "xfem_enriched"); 
    }
    data_out.add_data_vector (dist_solution, "xfem_solution");

    
    data_out.build_patches ();

    std::stringstream filename;
    filename << output_dir_ << "xmodel_dist_solution_" << cycle << ".vtk";
    
    std::ofstream output (filename.str());
    data_out.write_vtk (output);
    data_out.clear();
    
    std::cout << "\noutput written in:\t" << filename.str() << std::endl;
  
  
    if(output_options_ & OutputOptions::output_shape_functions)
    {
        unsigned int n_dofs = dof_handler->n_dofs();
        data_out.attach_dof_handler (dist_dof_handler);
        std::vector<Vector<double> > dist_dof_func;
        //writing only half of the enriched functions
        for(unsigned int i = n_dofs; i < n_enriched_dofs_/2+n_dofs; i++)
        {
        dist_dof_func.push_back(Vector<double>(dist_dof_handler.n_dofs()));
        get_dof_func(support_points,i, dist_dof_func.back() );
        DBGMSG("output of xfem shape function of enriched dof %d\n",i);
        }
    
        for(unsigned int i = 0; i < dist_dof_func.size(); i++)
        {
        std::stringstream func_name;
        func_name << "func_" << i+n_dofs;
        data_out.add_data_vector (dist_dof_func[i], func_name.str());
        }
        
        data_out.build_patches ();
    
        std::stringstream filename_x;
        filename_x << output_dir_ << "xshape_func" << ".vtk";
    
        std::ofstream output_x (filename_x.str());
        data_out.write_vtk (output_x);
    }
    
    //clearing data, releasing pointers (expecially to DofHandler)
    data_out.clear();
}


void XModel::output_distributed_solution(const std::string& mesh_file, const std::string &flag_file, bool is_circle, const unsigned int& cycle, const unsigned int &m_aquifer)
{ 
  //triangulation for distributing solution onto domain
  Triangulation<2> dist_coarse_tria;  
  PersistentTriangulation<2> *dist_tria;  
  
  //====================opening mesh
  //open filestream with mesh from GMSH
  std::ifstream in;
  GridIn<2> gridin;
  in.open(mesh_file);
  //attaching the read grid
  gridin.attach_triangulation(dist_coarse_tria);
  if(in.is_open())
  {
    //reading data from filestream
    gridin.read_msh(in);
  }          
  else
  {
    xprintf(Err, "Could not open grid file: %s", mesh_file.c_str());
  }
  
  //creating persistent triangulation
  dist_tria = new PersistentTriangulation<2>(dist_coarse_tria);
  
  //now we will try to load refine flags
  in.close();
  in.clear();
  in.open(flag_file);
  if(in.is_open())
  {
    //if flags file can be read
    dist_tria->read_flags(in);
    if(is_circle)
    {
      DBGMSG("Warning: Be sure that the area of the model is set - the circle triangulation will be computed of it.");
      Point<2> center((down_left+up_right)/2);
      double radius =  down_left.distance(up_right) / 2;
      static const HyperBallBoundary<2> boundary_load(center,radius);
      dist_coarse_tria.set_boundary(0, boundary_load);
    }
  }          
  else if (flag_file != "")
  {
    xprintf(Warn, "Could not open refinement flags file: %s\n Ingore this if loading mesh without refinement flag file.", flag_file.c_str());
  }
  
  //restore in both cases (flags or not)
  dist_tria->restore();
  
  std::cout << "...computing solution on:   " << mesh_file << std::endl;
  output_distributed_solution(*dist_tria, cycle, m_aquifer);
  
  //destroy persistent triangulation, release pointer to coarse triangulation
  delete dist_tria;
}


std::pair<double,double> XModel::integrate_difference(dealii::Vector< double >& diff_vector, ExactBase * exact_solution, bool h1)
{
    unsigned int m = n_aquifers_-1;
    MASSERT(triangulation != nullptr, "No triangulation in model.");
    triangulation->clear_user_data();
    triangulation->load_user_pointers(tria_pointers_[m]); //reload proper xdata_
    
  std::pair<double,double> norms;
  switch(enrichment_method_)
  {
    case Enrichment_method::xfem: 
        norms = integrate_difference<Enrichment_method::xfem>(diff_vector, exact_solution);
        break;
    case Enrichment_method::xfem_ramp: 
        norms = integrate_difference<Enrichment_method::xfem_ramp>(diff_vector, exact_solution);
        break;
      
    case Enrichment_method::xfem_shift:  
        norms = integrate_difference<Enrichment_method::xfem_shift>(diff_vector, exact_solution);
        break;
      
    case Enrichment_method::sgfem:
        norms = integrate_difference<Enrichment_method::sgfem>(diff_vector, exact_solution);
        break;
    default: 
        MASSERT(0,"Unknown enrichment type or not implemented.");
  }
  return norms;
}



void XModel::compute_interpolated_exact(ExactBase *exact_solution)
{
    Vector<double> unenriched(dof_handler->n_dofs());
    Vector<double> enriched(dof_handler->n_dofs());
    //Vector<double> solution(dof_handler->n_dofs());
    
    std::vector<unsigned int> local_dof_indices (fe.dofs_per_cell);   
    XDataCell *cell_xdata;
    
    DoFHandler<2>::active_cell_iterator
        cell = dof_handler->begin_active(),
        endc = dof_handler->end();
    for (; cell!=endc; ++cell)
    {
      local_dof_indices.resize(fe.dofs_per_cell);
      if(cell->user_pointer() == nullptr)
        cell->get_dof_indices(local_dof_indices);
      else
      {
        cell_xdata = static_cast<XDataCell*>(cell->user_pointer());
        cell_xdata->get_dof_indices(local_dof_indices ,fe.dofs_per_cell);
      }
//      std::cout << "at boundary: " << cell->at_boundary() << std::endl;
      for(unsigned int i=0; i<fe.dofs_per_cell; i++)
      {
        unenriched[local_dof_indices[i]] = exact_solution->value(cell->vertex(i));
        enriched[local_dof_indices[i]] = exact_solution->a();
        //solution[local_dof_indices[i]] = unenriched[local_dof_indices[i]] + enriched[local_dof_indices[i]];
        
//         std::cout << "exact a = " << unenriched[local_dof_indices[i]] 
//             << "\t computed a = " << block_solution[local_dof_indices[i]] 
//             << "\t diff = " << unenriched[local_dof_indices[i]] - block_solution[local_dof_indices[i]];
//          
//         if(cell->user_pointer() != nullptr)
//         {
//             if(i+fe.dofs_per_cell < local_dof_indices.size())
//             std::cout << "\t exact b = " << enriched[local_dof_indices[i]]
//                 << "\t computed b = " << block_solution[local_dof_indices[i+fe.dofs_per_cell]]  
//                 << "\t diff = " << enriched[local_dof_indices[i]] - block_solution[local_dof_indices[i+fe.dofs_per_cell]] << std::endl;
//             else std::cout << std::endl;
//         }
//         else std::cout << std::endl;
      }   
    }
    
//     std::cout << exact_solution->a() << std::endl;
//     for(unsigned int i=0; i<n_enriched_dofs; i++)
//     {
//         std::cout << setprecision(10) << block_solution.block(1)[i] << std::endl;
//     }
    
    //====================vtk output
    DataOut<2> data_out;
    data_out.attach_dof_handler (*dof_handler);
  
    hanging_node_constraints.distribute(unenriched);
    //hanging_node_constraints.distribute(enriched);
    //hanging_node_constraints.distribute(solution);
  
    //data_out.add_data_vector (unenriched, "exact_unenriched");
    //data_out.add_data_vector (enriched, "exact_enriched");
    data_out.add_data_vector (unenriched, "exact_solution");
    data_out.build_patches ();

    std::stringstream filename;
    filename << output_dir_ << "exact_solution_" << cycle_ << ".vtk";
   
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
    //*/
}