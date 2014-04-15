using DataFrames
using Gadfly
addprocs(3)
@everywhere include("hw5.jl")

function mainsvd2()
    const tol = 1e-4
    res = @parallel vcat for k in 1:8
        @parallel vcat for p in 0.6:0.1:1.0
            A, Ahat = genA(k,p)
            U,S,V = svdP(Ahat,k,tol)
            recon = U*S*V'
            fn = normfro(A-recon)
            info("computed: error: $fn, k: $k, p: $p")
            {:error => fn, :k => k, :p => p}
        end
    end
    df = DataFrame(res)
    writetable("svddata2-2.csv",df)
    p = plot(df,x="k",y="error", color="p",Geom.line)
    draw(PNG("svdplot.png",6inch,3inch),p)
    return df
end

mainsvd2()
