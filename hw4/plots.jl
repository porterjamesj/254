using Gadfly
using DataFrames

df = readtable("data1.csv")

# draw plot of log-likelihood data
p = plot(df,x="nstates",y="log_likelihood", Geom.line)
draw(PNG("llplot.png",6inch,3inch),p)
