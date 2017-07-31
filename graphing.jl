using LightGraphs
using Colors
using Compose
using Plots

blacklist = String["twitter.com", "facebook.com", "youtube.com", "instagram.com",
                "plus.google.com", "linkedin.com", "t.co", "itunes.apple.com",
                "pinterest.com", "flickr.com", "bit.ly", "play.google.com"]
    
function constructGraph(body; minsamp=1, giant=false, mut=false)
    s2i = Dict{String, Int}()
    i2s = Dict{Int, String}()
    count = 0
    
    for src in keys(body)
        if ~(src in blacklist)
            for dst in keys(body[src])
                if body[src][dst] > minsamp && ~(dst in blacklist) && (~mut || body[dst][src] > minsamp) && src != dst
                    if ~haskey(s2i, src)
                        count += 1
                        s2i[src] = count
                        i2s[count] = src
                    end
                    if ~haskey(s2i, dst)
                        count += 1
                        s2i[dst] = count
                        i2s[count] = dst
                    end
                end
            end
        end
    end

    G = Graph(count)
    for src in keys(body)
        if ~(src in blacklist)
            for dst in keys(body[src])
                if body[src][dst] > minsamp &&  ~(dst in blacklist) && (~mut || body[dst][src] > minsamp) && src != dst
                    add_edge!(G, s2i[src], s2i[dst])
                end
            end
        end
    end
    
    if ~giant
        return G, s2i, i2s
    else
        sg = sort(connected_components(G), by=size, rev=true)[1]
        ss2i = Dict{String, Int}()
        si2s = Dict{Int, String}()
        
        c = 0
        for i in sg
            c += 1
            ss2i[i2s[i]] = c
            si2s[c] = i2s[i]
        end

        H = Graph(size(sg, 1))

        for j in 1:c
            src = si2s[j]
            if src in keys(body)
                for dst in keys(body[src])
                    if body[src][dst] > minsamp && ~(dst in blacklist) && (~mut || body[dst][src] > minsamp) && src != dst
                        add_edge!(H, j, ss2i[dst])
                    end
                end
            end
        end
        
        return H, ss2i, si2s
    end
end

function reality(bias, x, key)
    biasnames = bias[2]
    bias = bias[1]
    if x in biasnames
        v = bias[find(biasnames .== x)[1], 4]
        if key == "fake"
            if isna(v)
                return 2
            else
                return 3
            end
        elseif key == "bias"
            if isna(bias[find(biasnames .== x)[1], 2])
                return 1
            elseif (bias[find(biasnames .== x)[1], 2] in ["L", "LC"])
                return 2
            elseif (bias[find(biasnames .== x)[1], 2] in ["R", "RC"])
                return 3
            end
        end
    end
    return 1
end

function plotGraph(bias, G, s2i, i2s, lx, ly; color="bias", path="plot/plot.png")
    colors = [colorant"grey", colorant"blue", colorant"red"]
    if color in ["fake", "bias"]
        members = [reality(bias, i2s[v], color) for v in 1:nv(G)]
        nodefillc = colors[members]
    else
        nodefillc = map(x-> RGB(1x, 0, 1-x), color./maximum(color))
    end
    sizes = [0.25 + log(length(all_neighbors(G, v))) for v in 1:nv(G)]
    plo = gplot(G, lx, ly, nodelabel=[i2s[v] for v in 1:nv(G)], nodelabelsize=sizes, nodefillc=nodefillc)
    draw(PNG(path, 60cm, 60cm), plo)
end

function makeFolds(G, folds)
    srand(42)
    v2f = Dict{Int, Int}()
    for v in 1:nv(G)
        v2f[v] = rand(1:folds)
    end
    return v2f
end

function makeBeliefs(bias, G, s2i, i2s, key; folds=nothing, fold=1)
    v2r = Dict{Int, Int}()
    for v in 1:nv(G)
        v2r[v] = reality(bias, i2s[v], key)
    end
    
    ϕ = ones((nv(G),2))./2.0
    for v in 1:nv(G)
        if folds == nothing || folds[v] != fold
            r = v2r[v]
            if r == 3
                ϕ[v,:] = [0.99, 0.01]
            elseif r == 2
                ϕ[v,:] = [0.01, 0.99]
            end
        end
    end
    return ϕ
end

function AUC(xs, ys)
    ps = Array{Float64}(size(xs, 1), 2)
    ps[:, 1] = xs
    ps[:, 2] = ys
    ps = sortrows(ps, by=x->(x[1],x[2]))
    s = 0
    
    for i in 1:size(ps, 1)-1
        s += (ps[i+1,1]-ps[i,1])*(ps[i,2]+ps[i+1,2])/2
    end
    return s
end

function getROC(bias, G, s2i, i2s, key, b; folds=nothing, fold=1)
    v2r = Dict{Int, Int}()
    for v in 1:nv(G)
        v2r[v] = reality(bias, i2s[v], key)
    end
    
    roc_x = []
    roc_y = []
    acc = []
    range = linspace(0, 1, 51)
    for cutoff in range
        score = zeros(Int, (2, 2))
        for v in 1:nv(G)
            if folds == nothing || folds[v] == fold
                r = v2r[v]
                if r == 3
                    if b[v, 1] > cutoff
                        score[1,1] += 1
                    else
                        score[2,1] += 1
                    end
                    elseif r == 2
                    if b[v, 1] > cutoff
                        score[1,2] += 1
                    else
                        score[2,2] += 1
                    end
                end
            end
        end
        tp = score[1,1]/sum(score[:,1])
        fp = score[1,2]/sum(score[:,2])
        append!(roc_x, fp)
        append!(roc_y, tp)
        append!(acc, (score[1,1]+score[2,2])/sum(score))
    end
    return roc_x, roc_y, acc, range
end

function findInc(bias, G, s2i, i2s, key, b; folds=nothing, fold=1)
    v2r = Dict{Int, Int}()
    for v in 1:nv(G)
        v2r[v] = reality(bias, i2s[v], key)
    end
    
    roc_x = []
    roc_y = []
    range = linspace(0, 0.25, 26)
    acc = []
    for space in range
        score = zeros(Int, (3, 2))
        for v in 1:nv(G)
            if folds == nothing || folds[v] == fold
                r = v2r[v]
                if r == 3
                    if b[v, 1] > 0.5+space
                        score[1,1] += 1
                    elseif b[v, 1] < 0.5-space
                        score[3,1] += 1
                    else
                        score[2,1] += 1
                    end
                    elseif r == 2
                    if b[v, 1] > 0.5+space
                        score[1,2] += 1
                    elseif b[v, 1] < 0.5-space
                        score[3,2] += 1
                    else
                        score[2,2] += 1
                    end
                end
            end
        end
        append!(acc, (score[1,1]+score[3,2])/(sum(score)-sum(score[2,:])))
    end
    return acc, range
end