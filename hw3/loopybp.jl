using Images
using Iterators

const DIRS = ((0,1),(0,-1),(1,0),(-1,0))

# a representation of the messages recieved at a particular node
type Messages
    data::Matrix{Float64}
end

maxdiff(m1::Messages, m2::Messages) = maximum(m1.data - m2.data)

function Base.show(io::IO,m::Messages)
    up = repr(vec(m[(1,0)]))
    down = repr(vec(m[(-1,0)]))
    right = repr(vec(m[(0,1)]))
    left = repr(vec(m[(0,-1)]))
    res = "       $up\n"
    res *= "$left     $right\n"
    res *= "       $down"
    println(io,res)
end

function Base.showcompact(io::IO,m::Messages)
    up = repr(vec(m[(1,0)]))
    down = repr(vec(m[(-1,0)]))
    right = repr(vec(m[(0,1)]))
    left = repr(vec(m[(0,-1)]))
    print(io,"msg[$up,$down,$left,$right]")
end

function Base.getindex(m::Messages,dir::(Int,Int))
    # this is just a way of mapping from direction
    # tuples to a backing length-4 vector
    return m.data[(3*dir[1] + dir[2] + 5) / 2,:]
end

function Base.setindex!(m::Messages,val,dir::(Int,Int))
    # this is just a way of mapping from direction
    # tuples to a backing length-4 vector
    m.data[(3*dir[1] + dir[2] + 5) / 2,:] = val
end

function Base.ones(::Type{Messages},dim...)
    ret = Array(Messages,dim)
    for i in 1:length(ret)
        ret[i] = Messages([[1.0,1.0,1.0,1.0] [1.0,1.0,1.0,1.0]])
    end
    return ret
end

type NodeField
    observed::Matrix{Int}
    # prev and curr contain the preveiously recieved and currently
    # being recieved messages for each node
    prev::Matrix{Messages}
    curr::Matrix{Messages}

    function NodeField(observed::Matrix)
        # we pad to avoid having to check boundaries
        prev = ones(Messages, size(observed,1)+2, size(observed,2)+2)
        padded_obs = padarray(observed,[1,1],-1)
        curr = deepcopy(prev)
        return new(padded_obs,prev,curr)
    end
end


# iterate the indices of all nodes in the field
# make sure we account for padding
nodes(nf::NodeField) = product(2:size(nf.observed,1)-1,
                               2:size(nf.observed,2)-1)


# account for padding in computing size
Base.size(nf::NodeField) = (size(nf.observed,1)-2,size(nf.observed,2)-2)


# the function ϕ as per the assignment
function ϕ(x::Int,y::Int,θ::Float64)
    if x==-1 || y==-1  # for the padding
        return 1
    else
        return x-y==0 ? 1+θ : 1-θ
    end
end


# message values
# guess is the guess for the value of the recieving node
# reciever is the direction of the reciever from the sender
# prev is the messages to the sending node in the previous round
# obs is the observed value of the sender
function messageval(guess::Int,
                    reciever::(Int,Int),
                    prev::Messages,
                    obs::Int,
                    θ::Float64)
    res = [1.0 1.0]
    for dir in DIRS
        res[1] = res[1] * prev[dir][1]
        res[2] = res[2] * prev[dir][2]
    end
    # we don't want to take the reciever into account, so divide it out
    res[1] /= prev[reciever][1]
    res[2] /= prev[reciever][2]
    # don't forget the message from the observed node
    res[1] *= ϕ(0,obs,θ)
    res[2] *= ϕ(1,obs,θ)
    # phi for our weight, summing over 1 and zero
    res[1] *= ϕ(0,guess,θ)
    res[2] *= ϕ(1,guess,θ)
    return sum(res)
end


function loopybp(field::NodeField, θ::Float64; maxt = 40)
    converged = false
    t = 1
    while !converged && t < maxt
        # we will iterate over each node and compute the
        # messages this node should be recieveing
        for node in nodes(field)
            # get messages from latent neighbors
            for dir in DIRS
                sender = (node[1] + dir[1], node[2] + dir[2])
                opp = (-dir[1],-dir[2])
                sender_prev_messages = field.prev[sender...]
                sender_obs_value = field.observed[sender...]
                msgz = messageval(0,opp,sender_prev_messages,
                                  sender_obs_value,θ)
                msgo = messageval(1,opp,sender_prev_messages,
                                  sender_obs_value,θ)
                total = msgo + msgz
                # normalize
                msgz = msgz / total
                msgo = msgo / total
                # write these to our messages
                field.curr[node...][dir] = [msgz msgo]
            end
        end

        # swap curr and prev
        tmp = field.prev
        field.prev = field.curr
        field.curr = tmp

        # check for convergence
        largest = -Inf
        for i in 1:length(field.curr)
            mdiff = maxdiff(field.curr[i],field.prev[i])
            if  mdiff > largest
                largest = mdiff
            end
        end
        if largest < 1e-2
            converged = true
        end
        t += 1
    end
    if !converged
        warn("loopybp did not converge")
    end
    return t
end


# pull off the marginals for a label from a nodefield
function marginals(field::NodeField, label::Int, θ::Float64)
    ret = ones(size(field))
    # slice out the actual nodes
    nodes = field.curr[2:end-1,2:end-1]
    obs = field.observed[2:end-1,2:end-1]
    for (i,messages) in enumerate(nodes)
        for j in 1:size(messages.data,1)
            ret[i] *= messages.data[j,label+1]
        end
        # remember to include the message from the
        # observed node
        ret[i] *= ϕ(label,obs[i],θ)
    end
    return ret
end

function denoise(noisy::Matrix, θ::Float64; maxt = 40)
    field = NodeField(noisy)
    t = loopybp(field,θ; maxt=maxt)
    field.curr[2:end-1,2:end-1]
    mz = marginals(field,0,θ)
    mo = marginals(field,1,θ)
    ret = Array(Int, size(field))
    for i in 1:length(ret)
        ret[i] = int(mo[i] > mz[i])
    end
    return (t,ret)
end


function noise(M::Matrix,μ::Float64)
    ret = copy(M)
    for i in 1:length(ret)
        if rand() < μ
            ret[i] = ret[i]==0 ? 1 : 0
        end
    end
    return ret
end
