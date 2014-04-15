using DataFrames
using Gadfly
include("hw5.jl")

function maineigs()
    const tols = [1e-6,1e-7,1e-8,1e-9]
    res = Dict[]
    for size in [10,100,300,500,700,1000,2000]
        for tol in tols
            A = randn(size,size)
            A = A'A
            t = eigP(A,1,tol)[3]
            push!(res,{:t => t,:n => size,:ε => tol})
            info("computed: $size, $tol: $t")
        end
    end
    df = DataFrame(res)
    writetable("eigdata.csv",df)
    # draw plots
    p = plot(df,x="n",y="t", xgroup="ε",Geom.subplot_grid(Geom.line),
            Scale.x_log10,Scale.y_log10)
    draw(PNG("eigplot.png",10inch,6inch),p)
    return df
end

maineigs()
