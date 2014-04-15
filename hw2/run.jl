using Gadfly
using DataFrames

addprocs(4) # we're going to crossvalidate in parallel
# load library
@everywhere require("perceptron.jl")
# load data
data = readdata("train2k.databw.35")
labels = vec(readlabels("train2k.label.35"))

# run linear perceptron in one batch
(weightvec, mistakes) = perceptron(data, labels)

# draw plot
p = plot(x=[1:length(mistakes[1])],y=mistakes[1],
         Guide.xlabel("Examples seen"),Guide.ylabel("Mistakes"))
draw(PNG("mistakes_linear.png",6inch,3inch),p)


# do cross validation for different values of sigma
σvals = [0.5,1,2,5,10,50]
crossvals = crossvalidate(8, data, labels, σvals)

# plot crossvals
crossp = plot(y=res,x=σvals,Geom.boxplot, Scale.x_discrete,
              Guide.xlabel("σ value"),Guide.ylabel("mistakes"))

draw(PNG("crossvalidate.png",6inch,3inch),crossp)

# do kernel perceptron with sigma = 5
(weightvec, mistakes) = perceptron(data, labels, makegaussian(5))
# draw plot
p = plot(x=[1:length(mistakes[1])],y=mistakes[1],
         Guide.xlabel("Examples seen"),Guide.ylabel("Mistakes"))
draw(PNG("mistakes_kernel.png",6inch,3inch),p)


# get testing data
testdata = readdata("test200.databw.35")

# use linear perceptron to predict testing data
(w, mistakes) = perceptron(data, labels; batches =2)
lin_predictions = predict(w,testdata)
writedlm("test200.label.linear",lin_predictions)

# use kernel perceptron to predict testing data
(c, mistakes) = perceptron(data, labels, makegaussian(5); batches =2)
kern_predictions = predict(makegaussian(5),c,data,testdata)
writedlm("test200.label.kernel",kern_predictions)
