#include "data_cell.hh"
#include "system.hh"
#include "well.hh"

using namespace dealii;

DataCellBase::DataCellBase(const DoFHandler<2>::active_cell_iterator &cell, 
                     Well* well, 
                     const unsigned int &well_index)
: cell_(cell)
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

const std::vector< double >& XDataCell::node_enrich_value(unsigned int local_well_index)
{
  MASSERT(local_well_index < n_wells_,"Index of well exceeded the size of the vector.");
  return node_enrich_values_[local_well_index];
}

void XDataCell::initialize()
{
  node_enrich_values_.resize(n_wells_);
  //initialization of enrichment function values at nodes
  for(unsigned int w=0; w < wells_.size(); w++)
  {
    node_enrich_values_[w].resize(n_vertices_);
    for(unsigned int i=0; i < global_enriched_dofs_[w].size(); i++)
    {
      //DBGMSG("w: %d, i: %d\n",w,i);
      node_enrich_values_[w][i] = wells_[w]->global_enrich_value(cell_->vertex(i));
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





      