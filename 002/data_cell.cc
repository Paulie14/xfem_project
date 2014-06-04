#include "data_cell.hh"
#include "system.hh"
#include "well.hh"

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
  n_wells_ = wells_.size();
  n_vertices_ = 4;
}

Well* DataCellBase::get_well(const unsigned int& local_well_index)
{
  MASSERT(local_well_index < n_wells_,"Index of well exceeded the size of the vector.");
  return wells_[local_well_index]; 
}

unsigned int DataCellBase::get_well_index(const unsigned int &local_well_index)
{
  MASSERT(local_well_index < n_wells_,"Index of well exceeded the size of the vector.");
  return wells_indices_[local_well_index];
}

unsigned int DataCellBase::get_well_dof_index(const unsigned int& local_well_index)
{
  MASSERT(local_well_index < n_wells_,"Index of well exceeded the size of the vector.");
  return well_dof_indices_[local_well_index];
}


void DataCellBase::set_well_dof_indices(const std::vector<unsigned int> &well_dof_indices)
{
  MASSERT(well_dof_indices.size() == n_wells_, "Sizes of vectors wells and well_dof_indices must be equal.");
  well_dof_indices_.clear();
  well_dof_indices_ = well_dof_indices;
}


void DataCellBase::add_data(Well* well, const unsigned int& well_index)
{
  wells_.push_back(well);
  wells_indices_.push_back(well_index);
  n_wells_++;
}




DataCell::DataCell(const DoFHandler< 2  >::active_cell_iterator &cell, 
                   Well* well, 
                   const unsigned int &well_index, 
                   const std::vector< const Point< 2 >* > &q_points)
: DataCellBase(cell, well, well_index)
{
  q_points_.push_back(q_points);
}

const std::vector< const dealii::Point< 2 >* >& DataCell::q_points(const unsigned int &local_well_index)
{
  MASSERT(local_well_index < wells_.size(),"Index of well exceeded the size of the vector.");
  return q_points_[local_well_index]; 
}


void DataCell::add_data(Well* well, 
                        const unsigned int &well_index, 
                        const std::vector< const Point< 2 >* > &q_points)
{
  wells_.push_back(well);
  wells_indices_.push_back(well_index);
  q_points_.push_back(q_points);
  n_wells_++;
}




XDataCell::XDataCell(const dealii::DoFHandler< 2  >::active_cell_iterator& cell, 
                     Well* well, 
                     const unsigned int& well_index, 
                     const std::vector< unsigned int > &enriched_dofs,
		     const std::vector<unsigned int> &weights)
: DataCellBase(cell, well, well_index)
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
: DataCellBase(cell, well, well_index)
{
  q_points_.push_back(q_points);
  global_enriched_dofs_.push_back(enriched_dofs);
  weights_.push_back(weights);
}

const std::vector< unsigned int >& XDataCell::global_enriched_dofs(const unsigned int& local_well_index)
{
  MASSERT(local_well_index < n_wells_,"Index of well exceeded the size of the vector.");
  return global_enriched_dofs_[local_well_index];
}

const std::vector< unsigned int >& XDataCell::weights(const unsigned int& local_well_index)
{
  MASSERT(local_well_index < n_wells_,"Index of well exceeded the size of the vector.");
  return weights_[local_well_index];
}

const std::vector< const dealii::Point< 2 >* >& XDataCell::q_points(const unsigned int& local_well_index)
{
  MASSERT(local_well_index < n_wells_,"Index of well exceeded the size of the vector.");
  if(local_well_index < q_points_.size())
    return q_points_[local_well_index]; 
  else
    return dummy_q_points_; //returning zero vector
}


unsigned int XDataCell::n_enriched_dofs()
{
  MASSERT(n_xdofs_ > 0, "Call get_dof_indices() before!");
  return n_xdofs_;
}

unsigned int XDataCell::n_enriched_dofs(unsigned int local_well_index)
{
  MASSERT(n_xdofs_ > 0, "Call get_dof_indices() before!");
  return n_enriched_dofs_[local_well_index];
}

unsigned int XDataCell::n_wells_inside()
{
  MASSERT(n_xdofs_ > 0, "Call get_dof_indices() before!");
  return n_wells_inside_;
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
  cell_->get_dof_indices(local_dof_indices);
  n_enriched_dofs_.resize(n_wells_);
  n_xdofs_ = 0;
  n_wells_inside_ = 0;
  //getting enriched dof indices and well indices
  for(unsigned int w = 0; w < n_wells_; w++)
  {   
    n_enriched_dofs_[w] = 0;
    for(unsigned int i = 0; i < n_vertices_; i++)
    {
      //local_dof_indices[dofs_per_cell+w*n_vertices+i] = xdata->global_enriched_dofs(w)[i];
      if(global_enriched_dofs_[w][i] != 0)
      {
        local_dof_indices.push_back(global_enriched_dofs_[w][i]);
        n_xdofs_++;
        n_enriched_dofs_[w]++;
      }
    }
    if(w < q_points_.size())
      if(q_points_[w].size() > 0)
      {
        n_wells_inside_++;
        local_dof_indices.push_back(well_dof_indices_[w]); //one more for well testing funtion
      }
  }
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


      