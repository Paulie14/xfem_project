
#ifndef Parameters_h
#define Parameters_h

/** @brief Old class used to set global parameters of the model.
 * Is hardly used now.
 */
class Parameters
{
  public:
    static double radius;                 ///< Radius of the well.
    static double pressure_at_top;        ///< Pressure at the top of the well.
    static double x_dec;                  ///< Shifting of the center of the well along x-axis.
    static double n_q_points;             ///< Number of quadrature points of the well.
        
    static double sqr;                    ///< Side of the square area.
    static double transmisivity;          ///< Transmisivity \f$ T \f$ of the aquifer.
    static double perm2fer;               ///< Permeability \f$ \sigma \f$ between the well and the aquifer.
    static double perm2tard;              ///< Permeability \f$ c \f$ of the well between two aquifers.
        
    static unsigned int start_refinement; ///< Initial level of refinement, used in grid creation.
    
    static float coarsing_percentage;     ///< Percentage of number of elements that should be coarsed during refinement procedure.
    static float refinement_percentage;   ///< Percentage of number of elements that should be refined during refinement procedure.
    
    static unsigned int cycle;            ///< Number of @p run calls.
};

#endif // Parameters_h