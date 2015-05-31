#include "global_setting_writer.hh"

#include "xquadrature_base.hh"
#include "xquadrature_cell.hh"
#include "xmodel.hh"
#include "model_base.hh"
#include "model.hh"

void GlobalSettingWriter::write_global_setting(ostream& stream)
{
    stream << "=============== Global Model Setting =============="
           << std::endl
           << "Maximal number of solver iterations = " << ModelBase::solver_max_iter_
           << std::endl
           << "Solver tolerance = " << ModelBase::solver_tolerance_
           << std::endl
           << "Output refinement tolerance criterion = "  << ModelBase::output_element_tolerance_
           << std::endl
           << "-----------------------------\n"
           << "Maximal Xquadrature refinement level = " << ModelBase::adaptive_integration_refinement_level_
           << std::endl
           << "Square refinement distance factor criterion = " <<  XQuadratureBase::square_refinement_criteria_factor_
           << std::endl
           << "-----------------------------\n"
//            << "XquadratureCell - alpha tolerance criterion = " << XModel::alpha_tolerance_
//            << std::endl
           << "XquadratureCell - tolerance criterion c_empiric = " << XQuadratureCell::c_empiric_
           << std::endl
           << "XquadratureCell - tolerance criterion p_empiric = " << XQuadratureCell::p_empiric_
           << std::endl
           << "-----------------------------\n"
           << "Use polar quadrature = " << (XModel::use_polar_quadrature_ ? "true" : "false")
           << std::endl
           << "Maximal level of refinement for polar quadrature = " << XModel::polar_refinement_level_
           << std::endl
           << "Well band width for polar quadrature = " << XModel::well_band_width_ratio_
           << std::endl
           << "N steps in angle phi in well quadrature = " << XModel::well_log_n_phi_
           << std::endl
           << "Gauss degree in well quadrature = " << XModel::well_log_gauss_degree_
           << std::endl
           << "===================================================" << std::endl;
}
