using Iterators

function readdata(fname)
    return (fname |> open |> readdlm |> transpose)
end

function readlabels(fname)
    return fname |> open |> readdlm
end


# linear perceptron
function perceptron(data, labels; batches = 1)
    n = size(data,2)
    w = zeros(size(data,1))
    batchmistakes = Vector{Int}[]
    t = 1
    for i in cycle(1:n)
        if t > n*batches
            break
        end
        if i == 1
            mistakes = Array(Int,n)
            push!(batchmistakes,mistakes)
            nmistakes = 0
        end
        yhat = dot(w,slice(data,:,i)) >= 0 ? 1 : -1
        if yhat == -1 && labels[i] == 1
            w = w + slice(data,:,i)
            nmistakes += 1
        elseif yhat == 1 && labels[i] == -1
            w = w - slice(data,:,i)
            nmistakes += 1
        end
        mistakes[i] = nmistakes
        t += 1
    end
    return (w, batchmistakes)
end

function kernsum(K::Matrix, c, data, i, t, n)
    sum = 0
    cin = 1
    for j in cycle(1:n)
        if cin > t
            break
        end
        sum += c[cin] * K[j,i]
        cin += 1
    end
    return sum
end

# precompute the kernel matrix to improve perf
# on multiple batches
function kernelmatrix(k, data)
    n = size(data,2)
    K = Array(Float64,(n,n))
    for i in 1:n
        for j in 1:i
            K[j,i] = k(slice(data,:,i),slice(data,:,j))
        end
    end
    return K
end

# kernel perceptron
function perceptron(data, labels, k; batches = 1)
    n = size(data,2)
    c = zeros(n*batches)
    K = kernelmatrix(k, data)
    nmistakes = 0
    batches = Vector{Int}[]
    t = 1
    for i in cycle(1:n)
        if t > length(c)
            break
        end
        if i == 1
            mistakes = Array(Int,n)
            push!(batches,mistakes)
            nmistakes = 0
        end
        res = kernsum(K, c, data, i, t, n)
        yhat = res >= 0 ? 1 : -1
        if yhat == -1 && labels[i] == 1
            c[t] = 1
            nmistakes += 1
        elseif yhat == 1 && labels[i] == -1
            c[t] = -1
            nmistakes += 1
        end
        mistakes[i] = nmistakes
        t += 1
    end
    return (c, batches)
end

function kernsum(k::Function, c, traindata, newdata, i, m)
    sum = 0.0
    for j in 1:m
        sum += c[j] * k(slice(traindata,:,j), slice(newdata,:,i))
    end
    return sum
end


# prediction function for weight vector from linear perceptron
function predict(w::Vector, newdata::Matrix)
    m = size(newdata,2)
    predictions = Array(Int,m)
    for i in 1:m
        predictions[i] = dot(w,slice(newdata,:,i)) >= 0 ? 1 : -1
    end
    return predictions
end


# prediction function for support vectors / kernel
function predict(k::Function, c, traindata, newdata)
    n = size(newdata,2)
    m = size(traindata,2)
    predictions = Array(Int,n)
    for i in 1:n
        res = kernsum(k, c, traindata, newdata, i, m)
        predictions[i] = res >= 0 ? 1 : -1
    end
    return predictions
end

# do two-way cross validation on
# a dataset, returning the total mistakes
function crossvalidate(data1::Matrix,
                       data2::Matrix,
                       labels1::Vector,
                       labels2::Vector,
                       σ::Real)
    g = (x,y) -> gaussian(x,y,σ)

    # get c vectors by training each way
    c1 = perceptron(data1, labels1, g)[1]
    c2 = perceptron(data2, labels2, g)[1]

    mistakes1 = predict(g, c1, data1, data2, labels2)
    mistakes2 = predict(g, c2, data2, data1, labels1)
    return mistakes1 + mistakes2
end


function crossvalidate(m::Int, data::Matrix, labels::Vector, σvals::Vector)
    # randomly partition data into two
    n = size(data,2)
    perm = randperm(n)
    perm1 = perm[1:n/2]
    perm2 = perm[n/2+1:end]
    data1 = data[:,perm1]
    labels1 = labels[perm1]
    data2 = data[:,perm2]
    labels2 = labels[perm2]

    # get initial results
    res = pmap(σ -> crossvalidate(data1, data2, labels1, labels2, σ), σvals)

    # do m more trials
    for i in 1:m-1
        perm = randperm(n)
        perm1 = perm[1:n/2]
        perm2 = perm[n/2+1:end]
        data1 = data[:,perm1]
        labels1 = labels[perm1]
        data2 = data[:,perm2]
        labels2 = labels[perm2]

        res = hcat(res,
                   pmap(σ -> crossvalidate(data1, data2, labels1, labels2, σ), σvals))
    end
    return res
end
