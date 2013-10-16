#ifndef DataCell_h
#define DataCell_h

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>

#include "well.hh"

/** @brief Base class for data distributed umong cells.
 * We need to distribute some data from wells umong the cells
 * These are pointers to @p Well objects, quadrature points of wells,
 * in case of XFEM additional enriched degrees of freedom.
 */
class DataCellBase
{
public:
  /** @brief Constructor.
   * @param cell iterator to cell which this data belongs to
   */
  DataCellBase(const dealii::DoFHandler<2>::active_cell_iterator &cell)
    : cell_(cell)
  {}
  
  /** @brief Constructor.
   * @param cell iterator to cell which this data belongs to
   * @param well is pointer to well which lies in the cell
   * @param well_index is index of the well in the global vector of wells in model class
   */
  DataCellBase(const dealii::DoFHandler<2>::active_cell_iterator &cell, 
               Well *well, 
               const unsigned int &well_index);
  
  ///Destructor
  virtual ~DataCellBase()
  {}
  
  /// Returns pointer to the cell which this data belong to.
  inline dealii::DoFHandler<2>::active_cell_iterator get_cell()
  { return cell_; }
  
  ///Returns number of wells comunicating with the cell
  inline unsigned int n_wells()
  { return wells_.size(); } 
  
  /// Returns pointer to one of the wells comunicating with the cell this data belong to.
  /**
   * @param local_well_index is local well index in the cell
   * @return pointer to well
   */
  Well* get_well(const unsigned int &local_well_index);
  
  /// Returns global index of the well.
  /** @param local_well_index is local well index in the cell
   */
  unsigned int get_well_index(const unsigned int &local_well_index);
  
  /// Returns global dof index of the well.
  /** @param local_well_index is local well index in the cell_
   */
  unsigned int get_well_dof_index(const unsigned int &local_well_index);
  
  /// Sets global well indices of the wells in the cell.
  /** @param well_dof_indices is vector of global well indices.
   */
  void set_well_dof_indices(const std::vector<unsigned int> &well_dof_indices);
  
  /// Adds new data to this object.
  /**
   * @param well is pointer to well which lies in the cell
   * @param well_index is index of the well in the global vector of wells in model class
   */
  virtual void add_data(Well *well, const unsigned int &well_index);
  
protected:
  ///iterator of the cell to which this data object belongs
  dealii::DoFHandler<2>::active_cell_iterator cell_;
  ///vector of pointers to wells
  std::vector<Well*> wells_;
  ///global indices of the wells
  std::vector<unsigned int> wells_indices_;
  ///global dof indices of the wells
  std::vector<unsigned int> well_dof_indices_;
};


//*************************************************************************************
//*************************************************************************************

/** @brief Class storing data from wells distributed to cells. Used in class @p Model.
 * 
 * This class is used to store data at the cell in the class @p Model.
 */
class DataCell : public DataCellBase
{
public:
  ///Constructor.
  /**
   * @param cell iterator to cell which this data belongs to
   */
  DataCell(const dealii::DoFHandler<2>::active_cell_iterator &cell) 
    : DataCellBase(cell)
  {}
  
  ///Constructor.
  /**
   * @param cell iterator to cell which this data belongs to
   * @param well is pointer to well which lies in the cell
   * @param well_index is index of the well in the global vector of wells in model class
   * @param q_points is vector of pointers to quadrature points of the well that lie in the cell
   * */
  DataCell(const dealii::DoFHandler<2>::active_cell_iterator &cell, 
            Well *well, 
            const unsigned int &well_index,
            const std::vector<const dealii::Point<2>* > &q_points);
  
  ///Destructor.
  virtual ~DataCell()
  {}
  
  ///Returns reference to vector of pointers to quadrature points of the well boundary
  const std::vector<const dealii::Point<2>* > &q_points(const unsigned int &local_well_index);

  /// Adds new data to this object.
  /**
   * @param well is pointer to well which lies in the cell
   * @param well_index is index of the well in the global vector of wells in model class
   * @param q_points is vector of pointers to quadrature points
   */
  void add_data(Well* well, 
                const unsigned int &well_index, 
                const std::vector<const dealii::Point<2>* > &q_points);
  
protected:
  ///each well has its own vector of quadrature points that lie on this cell
  std::vector< std::vector< const dealii::Point<2>* > > q_points_;
};



//*************************************************************************************
//*************************************************************************************

/** @brief Class storing data from wells distributed to cells. Used in class @p XModel.
 * 
 * This class stores similar data as class @p DataCell
 * but also the enriched degrees of freedom belonging to the current cell.
 */
class XDataCell : public DataCellBase
{
  public:
    /// Constructor.
    XDataCell(const dealii::DoFHandler<2>::active_cell_iterator &cell)
      : DataCellBase(cell)
    {}
    
    /// Constructor. 
    /// For using without quadrature points around the well edge.
    XDataCell(const dealii::DoFHandler<2>::active_cell_iterator &cell, 
              Well *well, 
              const unsigned int &well_index,
              const std::vector<unsigned int> &enriched_dofs,
              const std::vector<unsigned int> &weights);
    
    /// Constructor. 
    /// For using with quadrature points around the well edge.
    XDataCell(const dealii::DoFHandler<2>::active_cell_iterator &cell, 
              Well *well, 
              const unsigned int &well_index,
              const std::vector< unsigned int > &enriched_dofs,
              const std::vector<unsigned int> &weights,
              const std::vector<const dealii::Point<2>* > &q_points);
    
    /// Destructor
    virtual ~XDataCell()
    {}
    
    /// Getter for enriched dofs.
    const std::vector<unsigned int> &global_enriched_dofs(const unsigned int &local_well_index);
    
    /// Getter for weights.
    const std::vector<unsigned int> &weights(const unsigned int &local_well_index);
    
    /// Getter for quadrature points along the edge of the well.
    const std::vector<const dealii::Point<2>* > &q_points(const unsigned int &local_well_index);
    
    /// Add enriched data (without q_points)
    void add_data(Well *well, 
                  const unsigned int &well_index, 
                  const std::vector<unsigned int> &enriched_dofs,
                  const std::vector<unsigned int> &weights);
    
    /// Add enriched data
    void add_data(Well *well, 
                  const unsigned int &well_index, 
                  const std::vector<unsigned int> &enriched_dofs,
                  const std::vector<unsigned int> &weights,
                  const std::vector<const dealii::Point<2>* > &q_points);
    
  private:
    /** Global numbers of enriched DoFs. 
     * Index subset in \f$ \mathcal{M}_w \f$ (nodes on both reproducing and blending elements).
     * Access the index in format [well_index][local_node_index].
     */
    std::vector<std::vector<unsigned int> > global_enriched_dofs_;
    
    /** Weights of enriched nodes. 
     * Weight is equal \f$ g_u = 1 \$ at enriched node from subset \f$ \mathcal{N}_w \f$.
     * Weight is equal \f$ g_u = 0 \$ at enriched node from subset \f$ \mathcal{M}_w \f$ which is not in \f$ \mathcal{N}_w \f$
     * Access the index in format [well_index][local_node_index].
     */
    std::vector<std::vector<unsigned int> > weights_;
    
    /// Each well has its own vector of quadrature points that lie on this cell.
    std::vector< std::vector< const dealii::Point<2>* > > q_points_;
    
    ///just for returning zero lenght vector
    std::vector< const dealii::Point<2>* > dummy_q_points_; 
};

#endif // DataCell_h