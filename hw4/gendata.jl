using DataFrames

include("HMMs.jl")

function main()
    text = munge("Alice.txt")
    res = Dict[]
    for nstates in 1:27
        model = train(text,nstates)
        l = likelihood(text,model)
        push!(res,{"log_likelihood" => l.n,
                   "nstates" => nstates})
    end
    return res
end

res = main()
df = DataFrame(res)
writetable("data1.csv",df)
