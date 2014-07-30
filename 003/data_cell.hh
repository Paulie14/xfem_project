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
  { return n_wells_; } 
  
  inline int user_index() const {return user_index_;}
  inline void set_user_index(int index) {user_index_ = index;}
  inline void clear_user_index() {user_index_ = 0;}
  
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
  
  int user_index_;                      ///< Supplements the user index of cell.
  
  unsigned int n_wells_;                ///< Number of wells that affect the cell.
  unsigned int n_vertices_;             ///< Number of vertices.
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
    
    /// @name Getters
    //@{    
      /// Getter for enriched dofs by a single well.
      const std::vector<unsigned int> &global_enriched_dofs(const unsigned int &local_well_index);
      
      /// Getter for weights of a single well.
      const std::vector<unsigned int> &weights(const unsigned int &local_well_index);
      
      /// Getter for quadrature points along the edge of a single well.
      const std::vector<const dealii::Point<2>* > &q_points(const unsigned int &local_well_index);
      
      
      /** Getter for enrichment function value of a single well at nodes.
       * Provides acces to the map of node values of enrichment functions.
       */
      inline double node_enrich_value(unsigned int local_well_index, unsigned int local_vertex_index) const
      {
        return node_values->operator[](local_well_index).at(cell_->vertex_index(local_vertex_index));
      }
      
      /** Writes local DoFs in given vector: wells*[FE dofs, Xdofs, Wdofs]
       * Sets n_wells_inside, n_dofs, n_xdofs, n_wdofs.
       */
      void get_dof_indices(std::vector<unsigned int> &local_dof_indices, unsigned int fe_dofs_per_cell);
      
      /// Number of all degrees of freedom on the cell (from all wells).
      unsigned int n_enriched_dofs();
      
      /// Number of degrees of freedom on the cell (from a single wells).
      unsigned int n_enriched_dofs(unsigned int local_well_index);
      
      /// Number of wells that has nonzero cross-section with the cell.
      unsigned int n_wells_inside();
    //@}
    
    /// Add enriched data (without q_points).
    void add_data(Well *well, 
                  const unsigned int &well_index, 
                  const std::vector<unsigned int> &enriched_dofs,
                  const std::vector<unsigned int> &weights);
    
    /// Add enriched data (possibly with q_points).
    void add_data(Well *well, 
                  const unsigned int &well_index, 
                  const std::vector<unsigned int> &enriched_dofs,
                  const std::vector<unsigned int> &weights,
                  const std::vector<const dealii::Point<2>* > &q_points);
    
    /** STATIC function. Goes through given XDataCells objects and initialize node values of enrichment before system assembly.
     * @param data_vector is given output vector (by wells) of maps which map enrichment values to the nodes
     * @param xdata is given vector of XDataCell objects (includes enrichment functions and cells)
     * @param n_wells is the total number of wells in the model
     */
    static void initialize_node_values(std::vector<std::map<unsigned int, double> > &data_vector, 
                                       std::vector<XDataCell*> xdata, 
                                       unsigned int n_wells);
    
  private:
    
    /** Pointer to a vector that contains the computed values of enrichment functions at nodes.
     * It is filled by function @p initialize_node_values and the values are then accessed
     * by function @p node_enrich_value. 
     */
    std::vector<std::map<unsigned int, double> > *node_values;
    
    /** Global numbers of enriched DoFs. 
     * Index subset in \f$ \mathcal{M}_w \f$ (nodes on both reproducing and blending elements).
     * Access the index in format [well_index][local_node_index].
     */
    std::vector<std::vector<unsigned int> > global_enriched_dofs_; 
    
    std::vector<unsigned int> n_enriched_dofs_; ///<Number of enriched dofs by a single well.
    unsigned int n_xdofs_,                      ///< Total number of enriched dofs.
                 n_wells_inside_;               ///< Number of wells inside the cell.
    
    /** Weights of enriched nodes. 
     * Weight is equal \f$ g_u = 1 \$ at enriched node from subset \f$ \mathcal{N}_w \f$.
     * Weight is equal \f$ g_u = 0 \$ at enriched node from subset \f$ \mathcal{M}_w \f$ which is not in \f$ \mathcal{N}_w \f$
     * Access the index in format [local_well_index][local_node_index].
     */
    std::vector<std::vector<unsigned int> > weights_;
    
    /** Pointers to quadrature points of wells that lies inside the cell.
     * Access: Point = [local_well_index][q] 
     */
    std::vector< std::vector< const dealii::Point<2>* > > q_points_;
    
    ///just for returning zero lenght vector
    std::vector< const dealii::Point<2>* > dummy_q_points_;
    
};

#endif // DataCell_h