

using GraphPlot
using Plots
pyplot()
include("loadbuild.jl")
include("graphing.jl")
include("beliefprop.jl")
include("Psis.jl")

EDGEFILE = "data/save.feather"
BIASFILE = "data/bias.csv"
plotdir = "results/plots/"
problem_types = ["bias", "fake"]
epsilon = 0.34

if EDGEFILE == "data/save.feather"
  by_domain_flag = true
elseif EDGEFILE == "data/save2.feather"
  by_domain_flag = false
end

save = loadSave(EDGEFILE)

# size(unique(save[:sdom]))

#println(size(unique(vcat(save[:rsrc],save[:rdest]))))

bias = loadBias(BIASFILE)

cats = buildWeb(save, by_domain_flag) #only mutual links for "a"

edgeunion = countunion(cats["a"], cats["img"])
@time for x in ["link", "script"]
    countunion!(edgeunion, cats[x])
end

sets = [("union", 50)] #[("union",10)], [("mut", 1000), ("mut", 50), ("mut", 2), ("mut", 1), ("a", 1000), ("a", 500),("a", 100), ("img", 200), ("img", 50)]

#s2p = Dict{String, Array}()

roc_x = Vector{Float64}()
roc_y = Vector{Float64}()

k = 3 #number of folds

for (sn, set) in enumerate(sets)
   edgetype = set[1]
    if edgetype == "union"
      edgeset = edgeunion
    else
      edgeset = cats[set[1]]
    end

    # if ~by_domain_flag
    #   domainmap = Dict{String, String}()
    #   for j in edgeset
    #     domainmap[j.first] = split(j.first, '/')[1]
    #     for i in j.second
    #       domainmap[i.first] = split(i.first, '/')[1]
    #     end
    #   end
    # end

    G, s2i, i2s = constructGraph(edgeset, minsamp=set[2], giant=false, by_domain=by_domain_flag)
    #G2, s2i2, i2s2 = constructGraph(edgeset, minsamp=set[2], giant=true, by_domain=by_domain_flag)
    println(minimum([length([x for x in all_neighbors(G, v) if x != v]) for v in 1:nv(G)]))
    lx, ly = spring_layout(G)
    #lx2, ly2 = spring_layout(G2)

    for key in problem_types
        p = plot()
        plot!(p, [0, 1], [0, 1], line=:dash, label="")
        println(key*string(set))
        try mkdir(plotdir*string(set[1])) catch x end
        try mkdir(plotdir*string(set[1])*"/"*string(set[2])) catch x end
        mydir = plotdir*string(set[1])*"/"*string(set[2])*"/"

        # plotGraph(bias, G, s2i, i2s, lx, ly, color=key,
        #             path=mydir*key*"train.png")
        # ϕ = makeBeliefs(bias, G, s2i, i2s, key)
        # pr, b, lodds = beliefprop(G, ϕ, Psis(0.44), 1);
        # plotGraph(bias, G, s2i, i2s, lx, ly, color=b[:,1],
                    # path=mydir*key*"prop.png")

        # plotGraph(bias, G2, s2i2, i2s2, lx2, ly2, color=key,
        #             path=mydir*key*"train_main.png")
        # ϕ = makeBeliefs(bias, G2, s2i2, i2s2, key)
        # pr, b, lodds = beliefprop(G2, ϕ, Psis(0.44), 1);
        # plotGraph(bias, G2, s2i2, i2s2, lx2, ly2, color=b[:,1],
        #             path=mydir*key*"prop_main.png")

        folds = makeFolds(G, k) # makeFolds(G, k, i2s, domainmap)
        aucs = []
        for f in 1:k
          println("Fold " *string(f))
            ϕ = makeBeliefs(bias, G, s2i, i2s, key, folds=folds, fold=f)
            pr, b, lodds = beliefprop(G, ϕ, Psis(epsilon), 2);
            # for v in 1:nv(G)
            #     if folds[v] == f
            #         site = i2s[v]
            #         if ~haskey(s2p, site)
            #             s2p[site] = ones(1, size(sets, 1)+2)./2.0
            #             s2p[site][size(sets, 1)+1] = reality(bias, site, "bias")
            #             s2p[site][size(sets, 1)+2] = folds[v]
            #         end
            #         s2p[site][sn] = b[v]
            #     end
            # end

            roc_x, roc_y, acc, lvls, cms = getROC(bias, G, s2i, i2s, key, b, folds=folds, fold=f)
            auc = AUC(roc_x, roc_y)
            #println("Max "*string(maximum(acc))*" at "*string(lvls[find(x->x==maximum(acc), acc)[1]]))
            println("AUC : "*string(auc))
            best_cutoff = collect(keys(acc))[indmax(collect(values(acc)))]
            println("best cutoff threshold : "*string(best_cutoff))
            println("confusion matrix (best threshold) : ")
            show(cms[best_cutoff])
            println("")
            println("confusion matrix (0.5 threshold) : ")
            show(cms[0.5])
            println("")
            #println(typeof(roc_x))
            #println(typeof(roc_y))
            #println(roc_x)
            #println(roc_y)
            plot!(p, roc_x, roc_y,label="Fold $f (AUC=$(@sprintf("%.3f", auc)))")
            push!(aucs, auc)
        end
        avg_auc = sum(aucs)/k
        title!("Receiver Operating Characteristic (Average AUC=$(@sprintf("%.3f", avg_auc)))")
        println("Avg AUC : "*string(avg_auc))
        savefig(mydir*key*"_ROC_curve.png")

        open(mydir*"score.txt", "w") do f
            write(f, string(sum(aucs)/k))
        end
    end
end

# for site in keys(s2p)
#     s2p[site][10] = reality(bias, site, "bias")
# end

# s2p

# s2p["cnn.com"]
# s2p["bluebirdbanter.com"]

# using JSON
# open("metadata.json", "w") do f
#     write(f, JSON.json(s2p))
# end
