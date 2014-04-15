# script to produce data for BP analysis

using DataFrames

addprocs(4)

@everywhere include("loopybp.jl")

filenames = ["square.data", "disk.data", "trees-bw.data"]

thetas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
mus = [0.1,0.2,0.3,0.4,0.5]

function gatherdata(filenames, thetas, mus)
    return @parallel vcat for filename in filenames
        data = readdlm(filename,Int)
        @parallel vcat for mu in mus
            noised = noise(data,mu)
            @parallel vcat for theta in thetas
                res = denoise(noised,theta)
                time = res[1]
                denoised = res[2]
                error = sum(abs(data - denoised))/length(data)
                trial = {"file" => filename,
                         "error" => error,
                         "θ" => theta,
                         "μ" => mu,
                         "time" => time}
                info("run complete: file: $filename, error: $error, θ: $theta, μ: $mu")
                trial
            end
        end
    end
end

data = gatherdata(filenames, thetas, mus)
df = DataFrame(data)
writetable("data.csv",df)

# also write example images
@parallel for filename in ["square.data", "disk.data", "trees-bw.data"]
    data = readdlm(filename,Int)
    noised = noise(data,0.2)
    writedlm(filename * ".noised",noised)
    res = denoise(noised,0.5)
    denoised = res[2]
    writedlm(filename * ".denoised",denoised," ")
    info("run finished: $filename")
end
