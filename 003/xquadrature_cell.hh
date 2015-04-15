#ifndef XQUADRATURE_CELL_H
#define XQUADRATURE_CELL_H

#include "xquadrature_base.hh"

//forward declaration
namespace dealii{
        template<int,int> class Mapping;
}
template<int dim,int spacedim=dim> using DealMapping = dealii::Mapping<dim,spacedim>;

class Well;
class XDataCell;

class XQuadratureCell : public XQuadratureBase
{
public:
    /** Refinement methods.
    */
    struct Refinement
    {
        typedef enum {  edge,   ///< N levels of refinement according to a well edge
                        error,  ///< refinement governed by an error estimate
                        polar   ///< refinement up to the scale of a well, must be summed up with radial integration
        } Type;
    };
    
    XQuadratureCell(XDataCell * xdata,
                    const DealMapping<2> &mapping,
                    Refinement::Type type
                   );
    
    /// Creates refinement of a cell -- new quadrature.
    void refine(unsigned int max_level) override;
    
    /** @brief Calls gnuplot to create image of refined element.
     * 
      * Also can save the gnuplot script to file.
      * @param output_dir is the directory for output_dir
      * @param real is true then the element is printed in real coordinates
      * @param show is true then the gnuplot utility is started and plots the refinement on the screen
      */ 
    void gnuplot_refinement(const std::string &output_dir, bool real=true, bool show=false) override;
    
private:
    /// @brief Refinement along the well edge.
    /** If the square is crossed by the well edge
      * it will be refined.
      */
    bool refine_edge();
    
    /// @brief Refinement controlled by chosen tolerance.
    bool refine_error(double alpha_tolerance = 1e-2);
    
    /// @brief Refinement such that squares reach the scale of well.
    /** Additional integration in polar coordinates is supposed.
      */
    bool refine_polar();
    
    ///@name Refinement criteria.
    //@{
        /// Returns true if criterion is satisfied.
        /** Criterion: square diameter > C * (minimal distance of a node from well edge)
        */
        bool refine_criterion_a(Square &square, Well &well);
        
        /// Returns true if criterion is satisfied.
        /** Criterion: square diameter > C * (minimal distance of square from well center)
        */
//         bool refine_criterion_r_min(Square &square, double r_min);
        
        /// Returns number of nodes of @p square inside the @p well.
        unsigned int refine_criterion_nodes_in_well(Square &square, Well &well);
        
        /// Computes the alpha criterion for different n (quadrature order) and returns the quad. order
        unsigned int refine_criterion_alpha(double r_min);
        
        bool refine_criterion_h(Square &square, Well &well, double criterion_rhs);
        
        /// Computes r_min
        double compute_r_min(Square &square, unsigned int w);
    //@}

    void map_quadrature_points_to_real();
    
    /// XdataCell connected to current cell.
    XDataCell * xdata_;
    
    /// Mapping from real cell to unit cell.
    const DealMapping<2> *mapping_;
        
    /// Refinement method.
    Refinement::Type refinement_type_;
    
    /// Alpha tolerance set from outside.
    double alpha_tolerance_;
    
    /// Alpha in apriori adaptive criterion.
    static const std::vector<double> alpha_;
    
    /// Empiric constants for refine_error method.
    static const double c_empiric_, p_empiric_;
};

#endif  //XQUADRATURE_CELL_H