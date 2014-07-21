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
#set terminal x11
set terminal postscript eps enhanced color font 'Helvetica,10'
set output 'output.eps'

#set size ratio 0.5
set size square

set title 'Adaptive integration convergence graph'
set xlabel 'step'
set ylabel 'relative error'
#set xtics 0.1
#set ytics 0.1
set key top right
set xrange [1e-4:0.5]
set yrange [1e-8:1]
set log x 10
set log y 10
set format x "10^{%L}"
set format y "10^{%L}"
set grid y

eps = 1e-10
largest_step = 0.25

set lmargin at screen 0.1
set rmargin at screen 0.9
set bmargin at screen 0.1
set tmargin at screen 0.9

# ls linestyle, lt linetype, lc linecolor, lw linewidth
set style line 1 lt 2 lc rgb "black" lw 0.5
set style arrow 1 head nofilled size screen 0.01,9 ls 1

set multiplot
i_files = 0
filenames = "test_adaptive_integration2_p1/final_table.txt test_adaptive_integration2_p2/final_table.txt test_adaptive_integration2_p3/final_table.txt"
do for [file in filenames] {
b = largest_step
N = 8
i_files = i_files+1

do for [i=1:N] {
  a = b/2.
  
  #data fitting
  f(x) = k*x**h
  FIT_LIMIT = 1e-10
  k=0.5; h=2.0;
  fit [a-eps:b+eps] f(x) file using 4:3 via k,h

  g(x,a,b) = (x > a && x < b) ? f(x) : 1/0
  
  s = (a+b)/2.
  if(i_files == 1) {
    sxa = s-0.4*s
    sxb = f(s)+1.5*f(s)
    sya = 0.8*s
    syb = 0.8*f(s)
    set arrow 1 from sxa,sxb to sya,syb as 1
    set label 1 sprintf("%1.2f",h) at 0.95*sxa,1.1*sxb right
  }
  plot g(x,a,b) lt i_files title ''

  if (i == 1) {
        unset title
        unset xlabel
        unset ylabel
        unset border
        unset xtics
        unset ytics
  }
  b = a
} # end for i
  plot file using 4:3 title "error" with points lc i_files
  
} # end for file
unset multiplot



