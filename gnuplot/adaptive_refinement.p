# Gnuplot script for printing adaptively refined element.
# Made by Pavel Exner.
#
# Run the script in gnuplot:
# > load "g_script_adapt.p"
#
# Data files used:
# adaptive_integration_refinement_*.dat
# adaptive_integration_qpoints_*.dat
#
#
reset
set terminal x11
set size ratio -1
set key off
set parametric
set trange [0:2*pi]
fx0(t) = 5.43 + 0.2*cos(t)
fy0(t) = 5.43 + 0.2*sin(t)

# -1 list one file per line
ref_filelist = system("ls -1 adaptive_integration_refinement_*.dat")
qpoints_filelist = system("ls -1 adaptive_integration_qpoints_*.dat")

plot for [fn in ref_filelist] fn using 1:2 with lines lc rgb 'red',\
for [fn in qpoints_filelist]  fn using 1:2 with points lc rgb 'light-blue' pt 2,\
"elements" using 1:2 with lines lc rgb 'black',\
fx0(t),fy0(t) lc rgb 'blue'