#include "parameters.hh"

//geometry
double Parameters::radius = 0.02;
double Parameters::pressure_at_top = 2.0;

//x coordinate of the well
double Parameters::x_dec = 1.0;
//aquifer is square with side = 2*sqr        
double Parameters::sqr = 5.0;

double Parameters::n_q_points = 100;

double Parameters::transmisivity = 1.0;
double Parameters::perm2fer = 1e12;
double Parameters::perm2tard = 1e12;
       
//GRID
//starting refinement of the grid
unsigned int Parameters::start_refinement = 3;
//percentage of coarsing of the grid in each cycle
float Parameters::coarsing_percentage = 0.0;
//percentage of refinement of the grid in each cycle
float Parameters::refinement_percentage = 0.3;

//number of cycles during adapting process
unsigned int Parameters::cycle = 7;
