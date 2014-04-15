using LogFloats

difference(x,y) = x > y ? x-y : y-x

# test munging
# munge the text in the file
function munge(str::String;literal=false)
    if literal
        clean = str
    else
        all = readall(open(filename))
        clean = join(split(all),' ')
    end
    final = Int[]
    for char in clean
        if char == ' '
            push!(final,27)
        elseif char == '*'
            push!(final,0)  # zero indicates missing data
        else
            push!(final,int(char)-96)
        end
    end
    return final
end

# demunge lists of numbers back to strings
function demunge(data::Vector{Int})
    chars = Array(Char,length(data))
    for i in 1:length(data)
        if data[i] == 27
            chars[i] = ' '
        elseif data[i] == 0
            chars[i] = '*'
        else
            chars[i] = char(data[i]+96)
        end
    end
    return chars
end


# hmm type

type HMM{T <: FloatingPoint}
    nh::Int  # number of hidden states
    no::Int  # number of observed states
    t::Int   # training time
    π::Vector{T}  # size N_h so π is starting is hidden state i
    ω::Matrix{T}  # size N_h * N_o, so ω[i][j] is emitting j from i
    θ::Matrix{T}  # size N_h * N_h so θ[i][j] is transitioning from i to j

    HMM(nh,no,t,π,ω,θ) = new(nh,no,t,π,ω,θ)

    function HMM(nh::Int,no::Int,t::Int)
        π = rand(nh)
        π = π/sum(π)
        ω = rand(nh,no)
        for i in 1:size(ω,1)
            ω[i,:] = ω[i,:]/sum(ω[i,:])
        end
        θ = rand(nh,nh)
        for i in 1:size(θ,1)
            θ[i,:] = θ[i,:]/sum(θ[i,:])
        end
        new(nh,no,t,
            convert(Vector{T},π),
            convert(Matrix{T},ω),
            convert(Matrix{T},θ))
    end
end

Base.eltype{T}(::HMM{T}) = T

# alpha matrix, as per the lecture slides
# α[i][t] is α_{time t}(state i)
function alpha(observed::Vector{Int}, hmm::HMM)
    ret = Array(eltype(hmm),hmm.nh,length(observed))
    for state in 1:hmm.nh  # initialize all hidden states at time 1
        ret[state,1] = hmm.ω[state,observed[1]] * hmm.π[state]
    end
    for t in 2:length(observed)
        for state in 1:hmm.nh
            Σ = zero(eltype(hmm))
            for prevstate in 1:hmm.nh
                Σ += hmm.θ[prevstate,state] * ret[prevstate,t-1]
            end
            if observed[t] != 0
                tmp = hmm.ω[state,observed[t]] * Σ
            else
                tmp = Σ
            end
            ret[state,t] = tmp
        end
    end
    return ret
end


# beta matrix, as per the lecture slides
# β[i][t] is β_{time t}(state i)

function beta(observed::Vector{Int}, hmm::HMM)
    ret = zeros(eltype(hmm),hmm.nh,length(observed))
    for state in 1:hmm.nh
        ret[state,length(observed)] = one(eltype(hmm))
    end
    for t in reverse(1:length(observed)-1)
        for state in 1:hmm.nh
            for nextstate in 1:hmm.nh
                if observed[t] > 0
                    tmp = hmm.ω[state,observed[t]]
                else
                    tmp = 1.0
                end
                ret[state,t] += (hmm.θ[state,nextstate] *
                                 tmp *
                                 ret[nextstate,t+1])
            end
        end
    end
    return ret
end


# gamma[i,t] = state i ,time t
function gamma(α,β)
    ret = α .* β
    # sums[t] is sum at time t over all states
    sums = sum(ret,1)
    for i in 1:size(ret,1)
        for j in 1:size(ret,2)
            ret[i,j] = ret[i,j] / sums[j]
        end
    end
    return ret
end


# xi[t,i,j] is ξ_t(xt = i,x_{t+1} = j)
function xi(γ,α,β,hmm,observed)
    ret = Array(eltype(hmm),hmm.t-1,hmm.nh,hmm.nh)
    for t in 1:hmm.t-1  # we cannot comptue xi for the last timepoint
        for i in 1:hmm.nh
            for j in 1:hmm.nh
                ret[t,i,j] = (γ[i,t]/β[i,t] *
                              hmm.θ[i,j] *
                              hmm.ω[j,observed[t+1]] *
                              β[j,t+1])
            end
        end
    end
    return ret
end


# update pi in place and return whether pi has converged
function updatepi!(hmm,γ,delta=1e-2)
    maxdiff = convert(eltype(hmm),0.0)
    for state in 1:hmm.nh
        old = hmm.π[state]
        hmm.π[state] = γ[state,1]
        if difference(old,hmm.π[state]) > maxdiff
            maxdiff = difference(old,hmm.π[state])
        end
    end
    return maxdiff
end

# update omega in place and return max difference
function updateomega!(hmm,γ,observed,delta=1e-2)
    maxdiff = convert(eltype(hmm),0.0)
    # sum over all times for each state
    timesums = mapslices(sum,γ,[2])
    for emission in 1:hmm.no
        for state in 1:hmm.nh
            filteredsum = zero(eltype(hmm.ω))
            for t in 1:size(γ,2)
                if observed[t] == emission
                    filteredsum += γ[state,t]
                end
            end
            old = hmm.ω[state,emission]
            hmm.ω[state,emission] = filteredsum / timesums[state]
            if difference(old, hmm.ω[state,emission]) > maxdiff
                maxdiff = difference(old, hmm.ω[state,emission])
            end
        end
    end
    return maxdiff
end


function updatetheta!(hmm,ξ,γ)
    maxdiff = convert(eltype(hmm),0.0)
    # timesums[i,j] is sum over all t for transition from state i to state j
    ξsums = reshape(mapslices(sum,ξ,[1]),hmm.nh,hmm.nh)
    # gammasums[i] is sum over all t UP THROUGH T-1 for gamma at state i
    γsums = mapslices(sum,slice(γ,:,1:size(γ,2)-1),[2])
    for i in 1:hmm.nh
        for j in 1:hmm.nh
            old = hmm.θ[i,j]
            hmm.θ[i,j] = ξsums[i,j] / γsums[i]
            if difference(old, hmm.θ[i,j]) > maxdiff
                maxdiff = difference(old, hmm.θ[i,j])
            end
        end
    end
    return maxdiff
end


#
# train an HMM on the Int vector using nh hidden states,
# returning an HMM type representing the model learned
#
# note that the alphabet + ' ' should already have been munged into
# an integer vector s.t. 'a'=1,'b'=2,...,' '=27.
#
# text[t] represents the observed letter at time t (y_t)
#
function train!(text::Vector{Int},hmm::HMM;delta=1e-2,maxiters=100)
    converged = false
    t = 1
    while !converged && t < maxiters
        # compute alpha and beta
        α = alpha(text,hmm)
        β = beta(text,hmm)
        # compute gamma and zi
        γ = gamma(α,β)
        ξ = xi(γ,α,β,hmm,text)
        # update model parameters
        pimaxdiff = updatepi!(hmm,γ)
        omegamaxdiff = updateomega!(hmm,γ,text)
        thetamaxdiff = updatetheta!(hmm,ξ,γ)
        converged = (pimaxdiff < delta &&
                     omegamaxdiff < delta &&
                     thetamaxdiff < delta)
        info("difference: $pimaxdiff, $omegamaxdiff, $thetamaxdiff")
        t += 1
    end
    if converged == false
        warn("HMM training did not converge")
    else
        info("HMM training converged in $t iterations")
    end
end

function train(text::Vector{Int},nh::Int)
    const NUM_STATES = 27
    hmm = HMM{LogFloat}(nh,NUM_STATES,length(text))
    train!(text,hmm)
    return hmm
end


function likelihood(text::Vector{Int},hmm::HMM)
    T = hmm.t
    A = alpha(text, hmm)
    return sum(A[:,T])
end

# use a model to correct text with missing data
function correct(text::Vector{Int},hmm::HMM)
    α = alpha(text,hmm)
    β = beta(text,hmm)
    γ = gamma(α,β)
    letters = Int[]
    corrected = copy(text)
    for t in 1:length(corrected)
        if text[t] == 0
            likelihoods = zeros(eltype(hmm), hmm.no)
            # figure out which emission is most likely here
            for emission in 1:hmm.no
                tmp = 0.0
                for state in 1:hmm.nh
                    tmp += hmm.ω[state,emission] * γ[state,t]
                end
                likelihoods[emission] = tmp
            end
            corrected[t] = indmax(likelihoods)
            push!(letters,corrected[t])
        end
    end
    return (letters,corrected)
end

# lifted straight from StatsBase:
# https://github.com/JuliaStats/StatsBase.jl/blob/master/src/sampling.jl#L166-L177
function sample(wv::Vector)
    t = rand() * sum(wv)
    w = wv
    n = length(w)
    i = 1
    cw = w[1]
    while cw < t && i < n
        i += 1
        @inbounds cw += w[i]
    end
    return i
end

# produce a new chain of observations of length n
# from a trained HMM
function sample(hmm::HMM,n::Int)
    emissions = Array(Int,n)
    # first choose an initial hidden state
    current = sample(hmm.π)
    # now iterate until we have emitted n observations
    for i in 1:n
        # emit something
        emissions[i] = sample(vec(hmm.ω[current,:]))
        # transition to a new state
        current = sample(vec(hmm.θ[current,:]))
    end
    return emissions
end
