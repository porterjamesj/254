using Gadfly
using DataFrames
using Color

df = readtable("data.csv")

# draw plots for error
for filename in ["square.data","disk.data","trees-bw.data"]
    p = plot(df[df["file"] .== filename,:],x="θ",y="error",color="μ", Geom.line,
             Scale.ContinuousColorScale(p -> RGB(0,p,1)))
    draw(PNG(replace(filename,".data","") * ".png",6inch,3inch),p)
end

# draw plots for iterations
for filename in ["square.data","disk.data","trees-bw.data"]
    p = plot(df[df["file"] .== filename,:],x="θ",y="time",color="μ", Geom.line,
             Scale.ContinuousColorScale(p -> RGB(0,p,1)))
    draw(PNG(replace(filename,".data","") * "iters.png",6inch,3inch),p)
end
