module LogFloats

export logfloat, LogFloat, difference

immutable LogFloat <: FloatingPoint
    n::Float64
end

Base.convert(::Type{LogFloat},f::Float64) = LogFloat(log(f))
Base.convert(::Type{Float64},lf::LogFloat) = exp(lf.n)
Base.promote_rule(::Type{LogFloat},::Type{Float64}) = LogFloat

logfloat(x) = convert(LogFloat,x)
logfloat(x::AbstractArray) = copy!(similar(x,typeof(logfloat(one(eltype(x))))), x)
Base.one(x::LogFloat) = LogFloat(0.0)
Base.one(::Type{LogFloat}) = LogFloat(0.0)
Base.zero(x::LogFloat) = LogFloat(-Inf)
Base.zero(::Type{LogFloat}) = LogFloat(-Inf)


# a LogFloat(n) represents e^n
function Base.show(io::IO,lf::LogFloat)
    n = lf.n
    print(io, "e^$n")
end

function Base.showcompact(io::IO,lf::LogFloat)
    n = lf.n
    print(io, "e^$n")
end


function (*)(lf1::LogFloat,lf2::LogFloat)
    return LogFloat(lf1.n + lf2.n)
end

function (/)(lf1::LogFloat,lf2::LogFloat)
    return LogFloat(lf1.n - lf2.n)
end

function (+)(lf1::LogFloat, lf2::LogFloat)
    if lf1.n == -Inf && lf2.n == -Inf
        return logfloat(0.0)
    elseif lf1.n > lf2.n
        x = lf1.n
        y = lf2.n
    else
        x = lf2.n
        y = lf1.n
    end
    LogFloat(x + log1p(exp(y-x)))
end

# not accurate for small values and honey badger don't care
(-)(lf1::LogFloat, lf2::LogFloat) = logfloat(exp(lf1.n) - exp(lf2.n))

(<)(lf1::LogFloat, lf2::LogFloat) = lf1.n < lf2.n

end
