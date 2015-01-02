#include "well.hh"

Well::Well(double r, Point< 2 > cent)
: active_(true),
  radius_(r), 
  pressure_(0),
  pressure_set_(false),
  center_(cent)
{
    perm2aquifer_.resize(1);
    perm2aquitard_.resize(1);
}

Well::Well(double r, Point< 2 > cent, double perm2fer, double perm2tard)
: active_(true), 
  radius_(r), 
  pressure_(0),
  pressure_set_(false),
  center_(cent)
{
    perm2aquifer_.push_back(perm2fer);
    perm2aquitard_.push_back(perm2tard);
}

Well::Well(Well* well)
: active_(well->active_),
  radius_(well->radius_), 
  pressure_(well->pressure_),
  pressure_set_(well->pressure_set_),
  center_(well->center_), 
  perm2aquifer_(well->perm2aquifer_), 
  perm2aquitard_(well->perm2aquitard_)
{}



double Well::global_enrich_value(const Point< 2 > &point)
{
  double distance = point.distance(center_);
  if (distance <= radius_)
    return std::log(radius_);
  
  return std::log(distance);
}


Tensor< 1, 2 > Well::global_enrich_grad(const Point< 2 > &point)
{ 
  Tensor<1,2> grad; //initialize all entries with zero
  
  double distance = center_.distance(point);
  if (distance > radius_)
  {   
    distance = std::pow(distance,2);
    grad[0] = (point[0] - center_[0]) / distance;
    grad[1] = (point[1] - center_[1]) / distance;
  }
  return grad;  //returns zero if  (distance <= radius)
}

/*
//prints position and dofs
  void Well::print (const bool p_cells, const bool p_q_points)
  {
    
    std::vector<unsigned int> local_dof_indices(4);//dofs_per_cell);
    
    std::cout << "well(" << center_ << "):  dofs:" << dofs_per_cell <<  "\n";
    MASSERT(cells.size() > 0, "Vector of cells is empty");
    unsigned int point_counter = 0;
    
    if(p_cells)
    for(unsigned int c = 0; c < cells.size(); c++)
    {
      cells[c]->get_dof_indices (local_dof_indices);
      std::cout << "   cell(" << c << ")=" << cells[c]->index() << "  dofs=[ ";
      for (unsigned int i = 0; i < dofs_per_cell-1; i++)
      {
        std::cout << local_dof_indices[i] << ", ";
      }
      std::cout << local_dof_indices[dofs_per_cell-1] << " ]";
      
      //coordinates of vertices of degrees of freedom
      
      std::cout << "\tvertexes=[";
      for (unsigned int i = 0; i < dofs_per_cell-1; i++)
      {
        std::cout << cells[c]->vertex_index(i);
        std::cout << "(" << cells[c]->vertex(i) << ")" << ", ";
      }
      std::cout << cells[c]->vertex_index(dofs_per_cell-1) << "]";
      
      
      MASSERT(cells.size() == index_q_points.size(), "Vectors cells and index_q_points have to be of same size.");
      std::cout << "\tpoints: " << index_q_points[c].size() << "\n";
      point_counter += index_q_points[c].size();
      
      if(p_q_points)
      for (unsigned int i = 0; i < index_q_points[c].size(); i++)
      {
        std::cout << "\t" << index_q_points[c][i] 
                  << "\t" << q_points[index_q_points[c][i]] << std::endl;;
      }
    }
    std::cout << "\tpoint count = " << point_counter << std::endl;
  }
  */
  
void Well::evaluate_q_points(const unsigned int& n)
{
  q_points_.clear();
  q_points_.resize(n);
  
  //if we would need angles at some point
  //std::vector<double> phis(n);
  
  double phi = 2*M_PI / n;
  
  for(unsigned int i=0; i < n; i++)
  {
      q_points_[i] = Point<2>(center_[0]+radius_*std::cos(i*phi),center_[1]+radius_*std::sin(i*phi));
      //phis[i] = i*phi / M_PI * 180;
  }
}

