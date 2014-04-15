include("HMMs.jl")

pi = [.85,.15]
omega = [0.4 0.6
         0.5 0.5]

theta = [0.3 0.7
         0.1 0.9]

test = HMM{LogFloat}(2,2,4,pi,omega,theta)
observed = [1,2,2,1]
a = alpha(observed,test)
b = beta(observed,test)
g = gamma(a,b)
x = xi(g,a,b,test,observed)
