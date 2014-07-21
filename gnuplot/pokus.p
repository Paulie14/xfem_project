# Gnuplot script for printing adaptively refined element.
# Made by Pavel Exner.
#
# Run the script in gnuplot:
# > load "g_script_adapt.p"
#
# Data files used:
# adaptive_integration_refinement.dat
# adaptive_integration_qpoints.dat
#
#
reset
set terminal x11
#set size ratio 0.5
set size square

set title 'Convergence graph'
set xlabel 'refinement'
set ylabel 'error'
#set xtics 0.1
#set ytics 0.1
set key top right
set xrange [-5:5]
set yrange [-5:5]
#set logscale x
#set logscale y
set grid y
set sample 300

N = 7
f(x,a) = a*x

set lmargin at screen 0.1
set rmargin at screen 0.9
set bmargin at screen 0.1
set tmargin at screen 0.9

set multiplot
do for [i=1:5] {

    plot f(x,i) lt i title ''

    if (i == 1) {
        unset title
        unset xlabel
        unset ylabel
        unset border
        unset xtics
        unset ytics
    }
}
unset multiplot

