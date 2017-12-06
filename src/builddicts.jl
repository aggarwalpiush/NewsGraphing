
using Feather
using ProgressMeter
using JSON

df = Feather.read("save.feather")
size(df, 1)

counts = Dict{String, Dict}()
for key in ["a", "img", "script", "link"]
    counts[key] = Dict{String, Dict}()
end

p = Progress(size(df, 1), 0.1)
for i in 1:size(df, 1)
    sdom = get(df[i, 1])
    ddom = get(df[i, 2])
    key = get(df[i, 3])
    if isascii(sdom) && isascii(ddom)
        if ~haskey(counts[key], sdom)
            counts[key][sdom] = Dict{String, Int}()
        end
        if ~haskey(counts[key][sdom], ddom)
            counts[key][sdom][ddom] = 0
        end
        counts[key][sdom][ddom] += 1
    end
    next!(p)
end

open("save.json", "w") do f
    write(f, JSON.json(counts))
end


