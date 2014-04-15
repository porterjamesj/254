using DataFrames
using Gadfly
include("hw5.jl")

function mainsvd()
    const tols = [1e-6,1e-7,1e-8,1e-9,1e-10]
    res = Dict[]
    A = randn(10,20)
    for k in 1:10
        for tol in tols
            U,S,V = svdP(A,k,tol)
            recon = U*S*V'
            fn = normfro(A-recon)
            push!(res,{:error => fn, :k => k, :tol => tol})
            info("computed: error: $fn, k: $k, tol: $tol")
        end
    end
    df = DataFrame(res)
    writetable("svddata.csv",df)
    p = plot(df,x="k",y="error", xgroup="tol",Geom.subplot_grid(Geom.line))
    draw(PNG("svdplot.png",6inch,3inch),p)
    return df
end

mainsvd()
