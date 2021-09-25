set table "master.pgf-plot.table"; set format "%.5f"
set format "%.7e";; set datafile separator ","; plot for [i=4:104] "../data/exp_pro/pima/MSC_CIS/10/ll.csv" using 2:i with lines; 
