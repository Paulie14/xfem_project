# Gnuplot script for printing enrichment function
# Made by Pavel Exner.
#
#
set size ratio 0.5       #same scale on x and y

set terminal svg
set output "enrichment_func.svg"

unset title       #title off
unset key

unset x2tics
set border 1
unset ytics
unset y2tics


set style line 1 lt 1 lw 2 pt 3 linecolor rgb "black"
set style line 2 lt 1 lw 2 pt 3 linecolor rgb "grey"
set parametric
s = 1.0
set trange [0:s]
r = 0.1
x0(t) = (1-r)*t-s
x1(t) = (2*r)*t-r
x2(t) = (1-r)*t+r


plot -r, -t*log(r) ls 2,     r, -t*log(r) ls 2, x0(t),-log(abs(x0(t))) ls 1,      x1(t),0*x1(t)-log(r)  ls 1,      x2(t), -log(abs(x2(t))) ls 1    #xfem func
   
