function eigP(A::Matrix{Float64}, k::Int, tol::Float64)
    # first get random vector
    @assert tol < 0.01  # can't use higher tolernace than this
    @assert size(A,1) == size(A,2)  # A must be symmetric
    n = size(A,1)
    res = Array(Float64,n,k)
    firstconvergance = 0
    i = 1
    while i <= k
        r = randn(n)
        converged = false
        freq = 5
        t = 1
        while !converged
            if i > 1 && t % freq == 1
                freq = round(freq*1.25)
                for p in 1:i-1
                    prev = slice(res,:,p)
                    r = r - (dot(r,prev) * prev)
                end
            end
            newr = A*r
            newr = newr / norm(newr)
            eminus = norm(r - newr)
            eplus = norm(r + newr)
            converged = (eminus < tol) || (eplus < tol)
            if converged && any([norm(newr-res[:,p]) < 0.1 ||
                                 norm(newr+res[:,p]) < 0.1 for p in 1:i-1])
                #reset
                converged = false
                newr = randn(n)
                freq = 5
            end
            r = newr
            t += 1
        end
        if i==1
            firstconvergance = t
        end
        res[:,i] = r
        i += 1
    end
    # return the eignevectors, eigenvalues, and convergance time
    return (res, [rayleigh(A,res[:,i]) for i in 1:k],firstconvergance)
end

rayleigh(A,x) = dot(A*x,x) / dot(x,x)

eigP(fname::String, n, tol) = eigP(convert(Matrix{Float64},fname |> open |> readdlm),n,tol)


function svdP(A::Matrix{Float64},k::Int,tol::Float64)
    Asym = A'A
    V,S = eigP(Asym,k,tol)
    U = Array(Float64,size(A,1),k)
    for i in 1:size(V,2)
        v = V[:,i]
        Av = A*v
        U[:,i] = Av / norm(Av)
    end
    return (U,diagm(sqrt(S)),V)
end

svdP(fname::String,k,tol) = svdP(fname |> open |> readdlm,k,tol)

function genA(k::Int,p::Float64)
    const SIZE = 30
    # generate random singular vectors
    V = Array(Float64,SIZE,k)
    for i in 1:k
        r = randn(SIZE)
        V[:,i] = r/norm(r)
    end
    U = Array(Float64,SIZE,k)
    for i in 1:k
        r = randn(SIZE)
        U[:,i] = r/norm(r)
    end
    # generate random singular values
    S = rand(k) |> sort |> reverse |> diagm
    ret = V*S*U'
    @assert rank(ret) == k
    return (ret,fuzz(ret,p))
end

# return a fuzzed version of the Matrix in which
# every entry is replaced by 0 with probability = 1-p
function fuzz(A::Matrix{Float64},p::Float64)
    ret = copy(A)
    mu = mean(A)
    for i in 1:length(A)
        if rand() > p
            ret[i] = mu
        end
    end
    return ret
end


# extra credit - implementation of the Jacobi method

function indmax_nondiag(A::Matrix)
    curr = (1,2)
    for i in 1:size(A,1)
        for j in 1:size(A,2)
            if i != j && abs(A[i,j]) > abs(A[curr...])
                curr = (i,j)
            end
        end
    end
    return curr
end

function givensrot(i,j,theta,sz)
    G = eye(sz)
    c = cos(theta)
    s = sin(theta)
    G[i,i] = c
    G[j,j] = c
    if i > j
        G[i,j] = s
        G[j,i] = -s
    else
        G[j,i] = s
        G[i,j] = -s
    end
    return G
end

function jacobi(A::Matrix{Float64},tol::Float64)
    @assert size(A,1) == size(A,2)  # A must be symmetric
    S = A
    O = eye(A)
    converged = false
    while !converged
        j,i = indmax_nondiag(S)
        theta = 0.5*atan((2 * S[i,j]) / (S[j,j]-S[i,i]))
        G = givensrot(i,j,theta,size(S,1))
        Snew = G*S*G'
        Onew = G*O
        if (abs(dot(diag(S)/norm(S |> diag),diag(Snew)/norm(Snew |> diag)))+tol) >= 1
            converged = true
        end
        S = Snew
        O = Onew
    end
    # sort em
    S = diag(S)
    perm = sortperm(S)
    sort!(S)
    oldO = O'
    O = similar(O)
    for i in 1:length(perm)
        O[:,i] = oldO[:,perm[i]]
    end
    return (S,O)
end

function svdJ(A::Matrix{Float64},k::Int,tol::Float64)
    Asym = A'A
    V,S = jacobi(Asym,tol)
    V = V[:,1:k]
    S = S[1:k]
    U = Array(Float64,size(A,1),k)
    for i in 1:size(V,2)
        v = V[:,i]
        Av = A*v
        U[:,i] = Av / norm(Av)
    end
    return (U,diagm(sqrt(S)),V)
end


function gen_cooccurance(fname::String)
    const NMOVIES = 1682
    const NUSERS = 943
    res = zeros(NUSERS,NMOVIES)
    udata = readdlm(fname)
    for i in 1:size(udata,1)
        res[udata[i,1],udata[i,2]] = udata[i,3]
    end
    return res
end
