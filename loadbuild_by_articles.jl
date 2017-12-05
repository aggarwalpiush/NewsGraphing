using Feather
using DataFrames
using ProgressMeter

function loadSave()
    return Feather.read("save2.feather") #Feather.read("save.feather")
end

function loadBias()
    bias = readtable("bias.csv", header=false)
    code = r"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?"

    biasnames = String[]
    p = Progress(size(bias,1), 0.1)
    for i in 1:size(bias,1)
        if ~isna(bias[i, 5])
            url = match(code, AbstractString(bias[i, 5]))
            bias[i, 5] = replace(url[4], "www.", "")
            push!(biasnames, bias[i, 5])
        else
            push!(biasnames, "NA")
        end
    end
    return bias, biasnames
end

"""
    buildWeb(df)

Build a data structure for storing the links in an indexed form.

The first key is the type of link one of: ["a", "link", "script", "img", "mut"]
where a,link,script, and img are html tag types, and mut is "mutual a links".

df is a dataframe with columns sdom, ddom, type as the first three columns.
"""
function buildWeb(df)
    cats = Dict{String, Dict}()
    for key in ["a", "link", "script", "img", "mut"]
        cats[key] = Dict{String, Dict}()
    end

    p = Progress(size(df, 1), 0.1)
    for i in 1:size(df, 1)
        sdom = get(df[i, 4]) # rsrc
        ddom = get(df[i, 5]) # rdest #get(df[i, 2])
        if isascii(sdom) && isascii(ddom)
            cat = cats[get(df[i, 3])] # linktype
            if ~haskey(cat, sdom)
                cat[sdom] = Dict{String, Int}()
            end
            if ~haskey(cat[sdom], ddom)
                cat[sdom][ddom] = 1
            else
                cat[sdom][ddom] += 1
            end
        end
        next!(p)
    end

    p = Progress(length(cats["a"]), 0.1)
    for src in keys(cats["a"])
        for dst in keys(cats["a"][src])
            if haskey(cats["a"], dst) && haskey(cats["a"][dst], src)
                if ~haskey(cats["mut"], src)
                     cats["mut"][src] = Dict{String, Int}()
                end
                cats["mut"][src][dst] = cats["a"][src][dst]
            end
        end
        next!(p)
    end
    return cats
end
