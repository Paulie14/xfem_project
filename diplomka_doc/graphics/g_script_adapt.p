# Gnuplot script for printing adaptively refined element.
# Made by Pavel Exner.
#
# Data files used:
# adaptive_integration_refinement.dat
# adaptive_integration_qpoints.dat
#
#

 set terminal svg
# set output "adaptive_ref.svg"
 set output "adaptive_ref_detail.svg"
# set size square
set size 1,1
set origin 0,0

set nokey
#set yrange [-0.03:0.67]
#set xrange [4.95:5.67]

# for detail
set yrange [0.278:0.288]
set xrange [5.285:5.295]

set xtics 5.285, 0.002, 5.295
set ytics 0.278, 0.002, 0.288

set size ratio -1
set parametric
set trange [0:2*pi]
fx0(t) = 5.3 + 0.02*cos(t)
fy0(t) = 0.3 + 0.02*sin(t)
plot "adaptive_integration_refinement.dat" using 1:2 with lines,\
"adaptive_integration_qpoints.dat" using 1:2 with points lc rgb "light-blue",\
fx0(t),fy0(t)
