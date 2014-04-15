include("hw5.jl")

function main()
    infile = ARGS[1]
    k = int(ARGS[2])
    tol = float(ARGS[3])
    vecoutf = ARGS[4]
    valoutf = ARGS[5]
    @assert length(ARGS) == 5  "Not enough arguments!"
    eigvecs, eigvals = eigP(infile,k,tol)
    open(vecoutf,"w") do f
        writedlm(f,eigvecs)
    end
    open(valoutf,"w") do f
        writedlm(f,eigvals)
    end
end


main()
