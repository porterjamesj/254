include("hw5.jl")

function main()
    k = int(ARGS[1])
    p = float(ARGS[2])
    Af = ARGS[3]
    Ahatf = ARGS[4]
    @assert length(ARGS) == 4  "Not enough arguments!"
    A, Ahat = genA(k,p)
    open(Af,"w") do f
        writedlm(f,A)
    end
    open(Ahatf,"w") do f
        writedlm(f,Ahat)
    end
end


main()
