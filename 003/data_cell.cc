#include "data_cell.hh"
#include "system.hh"
#include "well.hh"

#include <deal.II/fe/mapping.h>

using namespace dealii;

DataCellBase::DataCellBase(const DoFHandler<2>::active_cell_iterator &cell, 
                     Well* well, 
                     const unsigned int &well_index)
: cell_(cell),
  user_index_(0)
{
  wells_.push_back(well);
  wells_indices_.push_back(well_index);
  well_dof_indices_.push_back(0);
  n_vertices_ = 4;
}

Well* DataCellBase::get_well(const unsigned int& local_well_index)
{
  MASSERT(local_well_index < wells_.size(),"Index of well exceeded the size of the vector.");
  return wells_[local_well_index]; 
}

unsigned int DataCellBase::get_well_index(const unsigned int &local_well_index)
{
  MASSERT(local_well_index < wells_.size(),"Index of well exceeded the size of the vector.");
  return wells_indices_[local_well_index];
}

unsigned int DataCellBase::get_well_dof_index(const unsigned int& local_well_index)
{
  MASSERT(local_well_index < wells_.size(),"Index of well exceeded the size of the vector.");
  return well_dof_indices_[local_well_index];
}

const std::vector< const Point< 2 >* >& DataCellBase::q_points(unsigned int local_well_index)
{
//     MASSERT(local_well_index < wells_.size(),"Index of well exceeded the size of the vector.");
//     return q_points_[local_well_index]; 
    MASSERT(local_well_index < wells_.size(),"Index of well exceeded the size of the vector.");
    if(local_well_index < q_points_.size())
        return q_points_[local_well_index]; 
    else
        return dummy_q_points_; //returning zero vector
}

const std::vector< Point< 2 > >& DataCellBase::mapped_q_points(unsigned int local_well_index)
{
    MASSERT(mapped_q_points_.size() > 0, "Quadrature points of the well have not been mapped yet. Call map_well_quadrature_points() at first!");
    MASSERT(local_well_index < wells_.size(), "Index of well exceeded the size of the vector.");
    
    return mapped_q_points_[local_well_index];
}

void DataCellBase::map_well_quadrature_points(const Mapping< 2 >& mapping)
{
    mapped_q_points_.clear();
    mapped_q_points_.resize(q_points_.size());
    for(unsigned int w=0; w < q_points_.size(); w++)
    {
        mapped_q_points_[w].resize(q_points_[w].size());
        for (unsigned int p =0; p < q_points_[w].size(); p++)
        {
            mapped_q_points_[w][p] = mapping.transform_real_to_unit_cell(cell_,*(q_points_[w][p]));
        }
    }
}

void DataCellBase::set_well_dof_indices(const std::vector<unsigned int> &well_dof_indices)
{
  MASSERT(well_dof_indices.size() == wells_.size(), "Sizes of vectors wells and well_dof_indices must be equal.");
  well_dof_indices_.clear();
  well_dof_indices_ = well_dof_indices;
}


void DataCellBase::add_data(Well* well, const unsigned int& well_index)
{
  wells_.push_back(well);
  wells_indices_.push_back(well_index);
}




DataCell::DataCell(const DoFHandler< 2  >::active_cell_iterator &cell, 
                   Well* well, 
                   const unsigned int &well_index, 
                   const std::vector< const Point< 2 >* > &q_points)
: DataCellBase(cell, well, well_index)
{
  q_points_.push_back(q_points);
}


void DataCell::add_data(Well* well, 
                        const unsigned int &well_index, 
                        const std::vector< const Point< 2 >* > &q_points)
{
  wells_.push_back(well);
  wells_indices_.push_back(well_index);
  q_points_.push_back(q_points);
}




XDataCell::XDataCell(const dealii::DoFHandler< 2  >::active_cell_iterator& cell, 
                     Well* well, 
                     const unsigned int& well_index, 
                     const std::vector< unsigned int > &enriched_dofs,
		     const std::vector<unsigned int> &weights)
:   DataCellBase(cell, well, well_index),
    n_enriched_dofs_(0),              
    n_wells_inside_(0),
    n_standard_dofs_(0),
    n_dofs_(0)
{
  global_enriched_dofs_.push_back(enriched_dofs);
  weights_.push_back(weights);
}


XDataCell::XDataCell(const dealii::DoFHandler< 2  >::active_cell_iterator& cell, 
                     Well* well, 
                     const unsigned int& well_index, 
                     const std::vector< unsigned int > &enriched_dofs,
                     const std::vector<unsigned int> &weights,
		     const std::vector< const dealii::Point< 2 >* >& q_points
		    )
:   DataCellBase(cell, well, well_index),
    n_enriched_dofs_(0),              
    n_wells_inside_(0),
    n_standard_dofs_(0),
    n_dofs_(0)
{
  q_points_.push_back(q_points);
  global_enriched_dofs_.push_back(enriched_dofs);
  weights_.push_back(weights);
}

const std::vector< unsigned int >& XDataCell::global_enriched_dofs(const unsigned int& local_well_index)
{
  MASSERT(local_well_index < wells_.size(),"Index of well exceeded the size of the vector.");
  return global_enriched_dofs_[local_well_index];
}

const std::vector< unsigned int >& XDataCell::weights(const unsigned int& local_well_index)
{
  MASSERT(local_well_index < wells_.size(),"Index of well exceeded the size of the vector.");
  return weights_[local_well_index];
}

double XDataCell::node_enrich_value(unsigned int local_well_index, unsigned int local_vertex_index) const
{
//  DBGMSG("N=%d, loc_w=%d, glob_w=%d, Nw=%d, loc_i=%d, i=%d\n",node_values->size(), local_well_index, wells_indices_[local_well_index], 
//         (*node_values)[local_well_index].size(),
//         local_vertex_index, cell_->vertex_index(local_vertex_index));
    return node_values->operator[](wells_indices_[local_well_index]).at(cell_->vertex_index(local_vertex_index));
}

unsigned int XDataCell::n_enriched_dofs(unsigned int local_well_index)
{
  MASSERT(n_enriched_dofs_ > 0, "Call get_dof_indices() before!");
  return n_enriched_dofs_per_well_[local_well_index];
}

unsigned int XDataCell::n_enriched_dofs()
{
  MASSERT(n_enriched_dofs_ > 0, "Call get_dof_indices() before!");
  return n_enriched_dofs_;
}

unsigned int XDataCell::n_wells_inside()
{
  MASSERT(n_enriched_dofs_ > 0, "Call get_dof_indices() before!");
  return n_wells_inside_;
}

unsigned int XDataCell::n_standard_dofs()
{
    MASSERT(n_standard_dofs_ > 0, "Call get_dof_indices() before!");
    return n_standard_dofs_;
}

unsigned int XDataCell::n_dofs()
{
    MASSERT(n_dofs_ > 0, "Call get_dof_indices() before!");
    return n_dofs_;
}



void XDataCell::initialize_node_values(std::vector<std::map<unsigned int, double> > &data_vector, 
                                       std::vector<XDataCell*> xdata, 
                                       unsigned int n_wells)
{
  data_vector.clear();
  data_vector.resize(n_wells);
  for(unsigned int k=0; k < xdata.size(); k++)
  {
    xdata[k]->node_values = &data_vector;
    for(unsigned int w=0; w < xdata[k]->n_wells(); w++)
    {
      for(unsigned int i=0; i < GeometryInfo<2>::vertices_per_cell; i++)
      {
        //if the map key 'xdata[k]->get_cell()->vertex_index(i)' does not exist, it creates new one
        data_vector[xdata[k]->get_well_index(w)][xdata[k]->get_cell()->vertex_index(i)] = 
          xdata[k]->get_well(w)->global_enrich_value( xdata[k]->get_cell()->vertex(i) );
      }
    }
  }
  DBGMSG("Number of node values: %d.\n", n_wells*data_vector[0].size());
}

void XDataCell::get_dof_indices(std::vector< unsigned int >& local_dof_indices, unsigned int fe_dofs_per_cell)
{
    local_dof_indices.resize(fe_dofs_per_cell);
    cell_->get_dof_indices(local_dof_indices);          //standard dofs
    n_enriched_dofs_per_well_.resize(wells_.size());
    n_enriched_dofs_ = 0;
    n_wells_inside_ = 0;
    //getting enriched dof indices and well indices
    for(unsigned int w = 0; w < wells_.size(); w++)          //enriched dofs
    {   
        n_enriched_dofs_per_well_[w] = 0;
        for(unsigned int i = 0; i < n_vertices_; i++)
        {
            //local_dof_indices[dofs_per_cell+w*n_vertices+i] = xdata->global_enriched_dofs(w)[i];
            if(global_enriched_dofs_[w][i] != 0)
            {
                local_dof_indices.push_back(global_enriched_dofs_[w][i]);
                n_enriched_dofs_++;
                n_enriched_dofs_per_well_[w]++;
            }
        }
    }   
    for(unsigned int w = 0; w < wells_.size(); w++)          //well dofs
    { 
        if(w < q_points_.size())
        if(q_points_[w].size() > 0)
        {
            n_wells_inside_++;
            local_dof_indices.push_back(well_dof_indices_[w]); //one more for well testing funtion
        }
    }
    n_standard_dofs_ = fe_dofs_per_cell;
    n_dofs_ = local_dof_indices.size();
}


void XDataCell::add_data(Well* well, 
                         const unsigned int& well_index, 
                         const std::vector< unsigned int >& enriched_dofs,
                         const std::vector<unsigned int> &weights)
{
  wells_.push_back(well);
  wells_indices_.push_back(well_index);
  global_enriched_dofs_.push_back(enriched_dofs);
  weights_.push_back(weights);
}


void XDataCell::add_data(Well* well, 
                         const unsigned int& well_index, 
                         const std::vector< unsigned int >& enriched_dofs,
                         const std::vector<unsigned int> &weights,
                         const std::vector< const dealii::Point< 2 >* >& q_points)
{
  wells_.push_back(well);
  wells_indices_.push_back(well_index);
  global_enriched_dofs_.push_back(enriched_dofs);
  weights_.push_back(weights);
  q_points_.push_back(q_points);
}


      