include("hw5.jl")

function main()
    infile = ARGS[1]
    k = int(ARGS[2])
    tol = float(ARGS[3])
    uout = ARGS[4]
    sout = ARGS[5]
    vout = ARGS[6]
    @assert length(ARGS) == 6  "Not enough arguments!"
    U,S,V = svdP(infile,k,tol)
    open(uout,"w") do f
        writedlm(f,U)
    end
    open(sout,"w") do f
        writedlm(f,S)
    end
    open(vout,"w") do f
        writedlm(f,V)
    end
end

main()
