# Gnuplot script for printing convergence lines.
# Made by Pavel Exner.
#
# Run the script in gnuplot:
# > load "xxx.p"
#
# Uses external file parameters.txt for fitting by linear regression.
# Uses multiplot to fit and plot several files, each by one regression.
#
reset
#set terminal x11
set terminal postscript eps enhanced color font 'Helvetica,15' linewidth 2
set output 'output.eps'

set size ratio 0.7
#set size square

set title 'Adaptive integration convergence graph'
set xlabel 'log(step)'
set ylabel 'log(relative error)'
#set xtics 0.1
#set ytics 0.1
set key top left
set xrange [1e-4:0.5]
set yrange [1e-7:0.01]
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
filenames = "test_adaptive_integration2_p1/final_table.txt test_adaptive_integration2_p2/final_table.txt test_adaptive_integration2_p3/final_table.txt test_adaptive_integration2_p4/final_table.txt"
do for [file in filenames] {
b = largest_step
N = 8
i_files = i_files+1

  
  #data fitting
  f(x) = k*x**h
  FIT_LIMIT = 1e-10
  k = 0.5; h = 2.0;
  fit f(x) file using 4:3 via k,h

  #set arrow 1 from sxa,sxb to sya,syb as 1
  #set label 1 sprintf("%1.2f",h) at 0.95*sxa,1.1*sxb right
    
  set key at 0.0002, 10**(-2 - 0.35*i_files)
  plot f(x) lt i_files title sprintf("k=%1.2f",h)

  if (i_files == 1) {
        unset title
        unset xlabel
        unset ylabel
        unset border
        unset xtics
        unset ytics
  }


  plot file using 4:3 title "" with points lc i_files
  
} # end for file
unset multiplot


