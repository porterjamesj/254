include("HMMs.jl")


text = munge("Alice.txt")
model = train(text,27)
res =sample(model,100) |> demunge |> join
println(res)
