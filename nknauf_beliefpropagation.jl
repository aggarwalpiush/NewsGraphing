using GraphPlot
using Plots
pyplot()
include("loadbuild.jl")
include("graphing.jl")
include("beliefprop.jl")
include("Psis.jl")

save = loadSave()

bias = loadBias()

cats = buildWeb(save)

G, s2i, i2s = constructGraph(cats["mut"], minsamp=10, giant=false)

sets = [("mut", 50)]#[("mut", 1000), ("mut", 50), ("mut", 2), ("mut", 1), ("a", 1000), ("a", 500),("a", 100), ("img", 200), ("img", 50)]

s2p = Dict{String, Array}()

roc_x = Vector{Float64}()
roc_y = Vector{Float64}()

for (sn, set) in enumerate(sets)
    G, s2i, i2s = constructGraph(cats[set[1]], minsamp=set[2], giant=false)
    G2, s2i2, i2s2 = constructGraph(cats[set[1]], minsamp=set[2], giant=true)
    println(minimum([length([x for x in all_neighbors(G, v) if x != v]) for v in 1:nv(G)]))
    lx, ly = spring_layout(G)
    lx2, ly2 = spring_layout(G2)
    for key in ["bias", "fake"]
        p = plot()
        plot!(p, [0, 1], [0, 1], line=:dash)
        println(key*string(set))
        try mkdir("plots/"*string(set[1])) catch x end
        try mkdir("plots/"*string(set[1])*"/"*string(set[2])) catch x end
        mydir = "plots/"*string(set[1])*"/"*string(set[2])*"/"

        plotGraph(bias, G, s2i, i2s, lx, ly, color=key,
                    path=mydir*key*"train.png")
        ϕ = makeBeliefs(bias, G, s2i, i2s, key)
        pr, b, lodds = beliefprop(G, ϕ, Psis(0.44), 10);
        plotGraph(bias, G, s2i, i2s, lx, ly, color=b[:,1],
                    path=mydir*key*"prop.png")

        plotGraph(bias, G2, s2i2, i2s2, lx2, ly2, color=key,
                    path=mydir*key*"train_main.png")
        ϕ = makeBeliefs(bias, G2, s2i2, i2s2, key)
        pr, b, lodds = beliefprop(G2, ϕ, Psis(0.44), 10);
        plotGraph(bias, G2, s2i2, i2s2, lx2, ly2, color=b[:,1],
                    path=mydir*key*"prop_main.png")

        folds = makeFolds(G, 3)
        aucs = []
        for f in 1:3
            ϕ = makeBeliefs(bias, G, s2i, i2s, key, folds=folds, fold=f)
            pr, b, lodds = beliefprop(G, ϕ, Psis(0.44), 2);

            for v in 1:nv(G)
                if folds[v] == f
                    site = i2s[v]
                    if ~haskey(s2p, site)
                        s2p[site] = ones(1, size(sets, 1)+2)./2.0
                        s2p[site][size(sets, 1)+1] = reality(bias, site, "bias")
                        s2p[site][size(sets, 1)+2] = folds[v]
                    end
                    s2p[site][sn] = b[v]
                end
            end

            roc_x, roc_y, acc, lvls = getROC(bias, G, s2i, i2s, key, b, folds=folds, fold=f)
            auc = AUC(roc_x, roc_y)
            #println("Max "*string(maximum(acc))*" at "*string(lvls[find(x->x==maximum(acc), acc)[1]]))
            println("AUC : "*string(auc))
            #println(typeof(roc_x))
            #println(typeof(roc_y))
            #println(roc_x)
            #println(roc_y)
            plot!(p, roc_x, roc_y)
            push!(aucs, auc)
        end
        savefig(mydir*key*"_ROC_curve.png")

        open(mydir*"score.txt", "w") do f
            write(f, string(sum(aucs)/3))
        end
    end
end

# for site in keys(s2p)
#     s2p[site][10] = reality(bias, site, "bias")
# end

# s2p

s2p["cnn.com"]
s2p["bluebirdbanter.com"]

# using JSON
# open("metadata.json", "w") do f
#     write(f, JSON.json(s2p))
# end
