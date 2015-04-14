#ifndef DataCell_h
#define DataCell_h

#include <deal.II/dofs/dof_accessor.h>

class XQuadratureWell;
//forward declarations
namespace dealii{
    template<int,int> class Mapping;
}

class Well;

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

    /// @name Getters
    //@{
    /// Returns pointer to the cell which this data belong to.
    dealii::DoFHandler<2>::active_cell_iterator get_cell();
    
    ///Returns number of wells comunicating with the cell
    unsigned int n_wells();
    
    int user_index() const;
    
    /// Returns pointer to one of the wells comunicating with the cell this data belong to.
    /**
     * @param local_well_index is local well index in the cell
     * @return pointer to well
     */
    Well* get_well(const unsigned int &local_well_index);
    
     /// Returns pointer to one of the wells comunicating with the cell this data belong to.
    /**
     * @param local_well_index is local well index in the cell
     * @return constant reference to a vector of pointers to wells
     */
    const std::vector<Well*> & get_wells();
    
    /// Returns global index of the well.
    /** @param local_well_index is local well index in the cell
     */
    unsigned int get_well_index(const unsigned int &local_well_index);
    
    /// Returns global dof index of the well.
    /** @param local_well_index is local well index in the cell_
     */
    unsigned int get_well_dof_index(const unsigned int &local_well_index);
    
    ///Returns reference to vector of pointers to quadrature points of the well boundary
    const std::vector<const dealii::Point<2>* > &q_points(unsigned int local_well_index);
    
    ///Returns reference to vector of pointers to quadrature points of the well boundary
    const std::vector<dealii::Point<2> > &mapped_q_points(unsigned int local_well_index);
    //@}
    
    void set_user_index(int index);
    void clear_user_index();
    
    /// Maps the quadrature points lying in the cell to a reference cell.
    void map_well_quadrature_points(const dealii::Mapping<2>& mapping);
    
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
    
    unsigned int n_vertices_;             ///< Number of vertices.
  
    /** Pointers to quadrature points of wells that lies inside the cell.
     * Access: Point = [local_well_index][q] 
     */
    std::vector< std::vector< const dealii::Point<2>* > > q_points_;
    
    /// Mapped well quadrature points (@p q_points_ on a reference cell).
    std::vector< std::vector<dealii::Point<2> > > mapped_q_points_;
    
    ///just for returning zero lenght vector
    std::vector< const dealii::Point<2>* > dummy_q_points_;
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
      
  /// Adds new data to this object.
  /**
   * @param well is pointer to well which lies in the cell
   * @param well_index is index of the well in the global vector of wells in model class
   * @param q_points is vector of pointers to quadrature points
   */
  void add_data(Well* well, 
                const unsigned int &well_index, 
                const std::vector<const dealii::Point<2>* > &q_points);
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
      
      
      /** Getter for enrichment function value of a single well at nodes.
       * Provides acces to the map of node values of enrichment functions.
       */
      double node_enrich_value(unsigned int local_well_index, unsigned int local_vertex_index) const;

      
      /** Writes local DoFs in given vector: wells*[FE dofs, Xdofs, Wdofs]
       * Sets n_wells_inside, n_dofs, n_xdofs, n_wdofs.
       */
      void get_dof_indices(std::vector<unsigned int> &local_dof_indices, unsigned int fe_dofs_per_cell);
      
      /// Number of all degrees of freedom on the cell (from all wells).
      unsigned int n_enriched_dofs();
      
      /// Number of degrees of freedom on the cell (from a single well).
      unsigned int n_enriched_dofs(unsigned int local_well_index);
      
      /// Number of wells that has nonzero cross-section with the cell.
      unsigned int n_wells_inside();
      
      /// Number of all degrees of freedom on the cell.
      unsigned int n_standard_dofs();
      
      /// Number of all degrees of freedom on the cell.
      unsigned int n_dofs();
      
      /// Number of polar quadratures for wells.
      unsigned int n_polar_quadratures(void);
      
      XQuadratureWell * polar_quadrature(unsigned int local_well_index);
      std::vector<XQuadratureWell *> polar_quadratures(void);
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
    
    void set_polar_quadrature(XQuadratureWell* xquad);
    
    void clear_polar_quadratures(void);
    
    /** STATIC function. Goes through given XDataCells objects and initialize node values of enrichment before system assembly.
     * @param data_vector is given output vector (by wells) of maps which map enrichment values to the nodes
     * @param xdata is given vector of XDataCell objects (includes enrichment functions and cells)
     * @param n_wells is the total number of wells in the model
     */
    static void initialize_node_values(std::vector<std::map<unsigned int, double> > &data_vector, 
                                       std::vector<XDataCell*> xdata, 
                                       unsigned int n_wells);
    
  private:
    
    /** Pointer to a vector that contains the precomputed values of enrichment functions at nodes.
     * It is filled by function @p initialize_node_values and the values are then accessed
     * by function @p node_enrich_value. 
     */
    std::vector<std::map<unsigned int, double> > *node_values;
    
    /// Quadratures in polar coordinates in vicinity of wells affecting the current cell.
    std::vector<XQuadratureWell*> well_xquadratures_;
    
    /** Global numbers of enriched DoFs. 
     * Index subset in \f$ \mathcal{M}_w \f$ (nodes on both reproducing and blending elements).
     * Access the index in format [well_index][local_node_index].
     */
    std::vector<std::vector<unsigned int> > global_enriched_dofs_; 
    
    std::vector<unsigned int> n_enriched_dofs_per_well_; ///<Number of enriched dofs by a single well.
    unsigned int n_enriched_dofs_,              ///< Number of all enriched dofs.
                 n_wells_inside_,               ///< Number of wells inside the cell.
                 n_standard_dofs_,              ///< Number of standard dofs.
                 n_dofs_,                       ///< Total number of dofs.
                 n_polar_quadratures_;          ///< Number of polar quadratures for wells.
    
    /** Weights of enriched nodes. 
     * Weight is equal \f$ g_u = 1 \$ at enriched node from subset \f$ \mathcal{N}_w \f$.
     * Weight is equal \f$ g_u = 0 \$ at enriched node from subset \f$ \mathcal{M}_w \f$ which is not in \f$ \mathcal{N}_w \f$
     * Access the index in format [local_well_index][local_node_index].
     */
    std::vector<std::vector<unsigned int> > weights_;
};










/****************************************            Implementation          ********************************/

inline dealii::DoFHandler<2>::active_cell_iterator DataCellBase::get_cell() { return cell_; }

inline unsigned int DataCellBase::n_wells() { return wells_.size(); } 

inline int DataCellBase::user_index() const {return user_index_;}

inline void DataCellBase::set_user_index(int index) {user_index_ = index;}

inline void DataCellBase::clear_user_index() {user_index_ = 0;}

inline const std::vector< Well* >& DataCellBase::get_wells() { return wells_;}

#endif // DataCell_h