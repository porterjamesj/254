using Images

# the euclidean distance between two vectors
function distance(x,y)
    @assert length(x) == length(y)
    res = 0
    for i = 1:length(x)
        res += (x[i]-y[i])^2
    end
    return sqrt(res)
end

function assign(point,centers)
    @assert size(centers)[1] == length(point)
    distances = Array(Float64,size(centers,2))
    for i = 1:size(centers,2)
        distances[i] = distance(point,centers[:,i])
    end
    return indmin(distances)
end

function center(assignments,data,i)
    # sanity check
    @assert length(assignments) == size(data,2) * size(data,3)

    # get indices of all points in the cluster i
    inds = find(x->x==i,assignments)
    sum = zeros(length(data[:,1]))
    for j in inds
        sum += data[:,j]
    end

    return sum / length(inds)
end

# cluster the given image
function kmeans(im::Image, k::Int; max_itr = Inf, tol = 1e-3)
    data = im.data
    # the number of pixels in the array
    pixels = size(data,2) * size(data,3)
    # create an array to hold the assignments
    assignments = Array(Int,pixels)

    # allocate space for the centers
    centers = zeros(Float64,3,k)
    # initialize the centers
    perm = randperm(pixels)
    for i = 1:k
        centers[:,i] = data[:,perm[i]]
    end


    # now run kmeans until convergance
    iterations = 0
    converged = false
    while !converged && iterations < max_itr
        # loop over all pixels and assign each to a cluster
        for i = 1:pixels
            assignments[i] = assign(data[:,i],centers)
        end

        differences = Array(Float64,size(centers)[2])
        # now recalculate the centers
        for i = 1:size(centers)[2]
            newcenter = center(assignments,data,i)
            differences[i] = distance(centers[:,i],newcenter)
            centers[:,i] = newcenter
        end

        if !bool(findfirst(x->x>tol,differences))
            converged = true
        end

        iterations += 1
    end

    return (assignments,centers)
end

# given an image, return a version of it
# k clusters, replacing each pixel by it's
# cluster center
function compress(im::Image,k::Int; max_itr = Inf, tol = 10e-3)
    ret = deepcopy(im)
    pixels = size(im.data,2) * size(im.data,3)
    (assignments, centers) = kmeans(im,k,max_itr=max_itr,tol=tol)

    # replace each pixel by it's cluster center
    for i = 1:pixels
        ret.data[:,i] = iround(centers[:,assignments[i]])
    end

    return ret
end
