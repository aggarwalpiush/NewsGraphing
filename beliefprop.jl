using Iterators

function normalize_neighbors!(m, g::Graph, X::AbstractArray, i::Integer)
    Z = sum(m[i,neighbors(g,i), :])
    #@show i, Z
    for j in neighbors(g,i)
        for k in X
            m[i,j,k] /= Z
        end
    end
    return Z
end

function assert_nonzero(val)
    @assert val != 0 "val was zero"
    return val
end

function propogate!{T<:Real, U<:Integer}(m::AbstractArray{T,3},
                                         ϕ::AbstractArray{T,2},
                                         ψ::AbstractArray{T,2}, A, X::AbstractArray{U, 1})
    n = size(A, 1)
    for i in 1:n
        for j in neighbors(A,i)
            if j == i
                continue
            end
            for k in X
                m[i,j,k] = 0
                for l in X
                    w = ϕ[i,l] * ψ[l,k]
                    @assert w != NaN
                    sp = sum([p!=j ? log(assert_nonzero(m[p,i,l])) : 0 for p in neighbors(A, i)])
                    p = exp(sp) + 1e-16
                    if p == 0
                        warn("sp is $sp")
                        warn("p is 0! $i,$j,$k,$l")
                    end
                    @assert p != NaN
                    @assert m[j,i,l] != 0 "$i,$j,$l have m 0"
                    m[i,j,k] += w * p #/ m[j,i,l]
                end
            end
        end
        Z = normalize_neighbors!(m, A, X, i)
    end
    return m
end

function beliefs!{T<:Real}(m::AbstractArray{T}, b, ϕ, A, X)
    for i in 1:size(A,1)
        for k in X
            w = prod(T[m[p,i,k] for p in neighbors(A, i)])
            if w == 0
                w = 1e-16
            end
            b[i,k] = ϕ[i,k] * w
            @assert b[i,k] >= 0 "Unnormalized beliefs are not all positive offending indices ($i, $k)=$(b[i,k])"
        end
        b[i,:] /= sum(b[i,:])
    end
    #@show b
    @assert all( 0.99 .< sum(b,2) .< 1.01) "Beliefs do not form a probability distributions"
    return b
end

type BeliefProblem{T<:Real, U<:Integer, G}
    m::AbstractArray{T,3}
    ϕ::AbstractArray{T,2}
    ψ::AbstractArray{T,2}
    A::G
    X::AbstractArray{U, 1}
end

type Messages{T<:Real, G} <: AbstractArray{T, 3}
    nrows::Int
    ncols::Int
    edges::Dict{Tuple{Int, Int}, Int}
    states::UnitRange
    storage::Array{T, 2}
end

function Messages(g::Graph, states::UnitRange)
    n=nv(g)
    d = Dict{Tuple{Int,Int}, Int}()
    i = 1
    for e in edges(g)
        d[(src(e), dst(e))] = i
        d[(dst(e), src(e))] = i+1
        i += 2
    end
    storage = ones(Float64, (i, length(states)))
    return Messages{Float64, Graph}(n,n,d, states, storage)
end

Base.size(m::Messages) = (m.nrows, m.ncols, length(m.states))
function Base.getindex(m::Messages, I::Vararg{Int})
    pair = I[1:2]
    if pair in keys(m.edges)
        return m.storage[m.edges[pair], I[3]]
    else
        throw(KeyError(pair))
    end
end

function Base.setindex!(m::Messages, v, I::Vararg{Int})
    pair = I[1:2]
    if pair in keys(m.edges)
        return m.storage[m.edges[pair], I[3]] = v
    else
        throw(KeyError(pair))
    end
end

logodds(b::AbstractMatrix) = log.(b[:,1]./b[:,2])

function beliefprop(g::Graph, ϕ, ψ, maxiter)
    n = size(g, 1)
    k = 2
    m = ones(Float64, (n,n,k))
    m = Messages(g, 1:k)
    #@show sort(collect(keys(m.edges)))
    pr = BeliefProblem(m, ϕ, ψ, g, 1:k)
    iterstates = []
    diffs = []
    i = 1
    m = nth(iterate(m->begin msg = propogate!(m, ϕ,ψ, g, 1:k);
            state = [maximum(msg.storage), minimum(msg.storage)]
            diff = i > 1 ? state-iterstates[i-1]:0;
            i+=1
            #@show diff
            push!(iterstates, state)
            push!(diffs, diff)
            #msg.storage = max(msg.storage, 1e-60)
           return msg 
        end, m), maxiter)
    b = similar(pr.ϕ)
#     for i in 1:n
#         if sum(pr.ϕ[i,:]) != sum(b[i, :])
#             println("Crud! "*string(pr.ϕ[i,:])*" "*string(b[i,:]))
#         end
#     end
    beliefs!(pr.m, b, pr.ϕ, g, 1:k)
    lodds = logodds(b)
    #plotbeliefs(g, b[:,1])
    #display(bar(lodds, ylabel="Log Odds", xlabel="Vertex ID"))
    return pr, b, lodds
end

function plotbeliefs(g::Graph, b::AbstractVector)
    #nodefillc=map(x-> RGB(1x, 0, 1-x), b./maximum(b))
    #plo = gplot(H, lx, ly, nodelabel=[si2s[v] for v in 1:nv(g)], nodelabelsize=sizes, nodefillc=nodefillc)
    #draw(PNG("plots/postbeliefprop.png", 60cm, 60cm), plo)
    #return plo
    #return gplot(g, layout=(g)-> spring_layout(g; C=1), nodelabel=1:nv(g), nodelabelsize=1, nodefillc=nodefillc)
    #return gplot(g, layout=(g)-> begin pos = (200*begin Z=spring_layout(g); Z[1]end, log2(float(collect(1:nv(g))))); @show typeof(pos); return pos end, nodelabel=1:nv(g), nodelabelsize=1, nodefillc=nodefillc)
    #return gplot(g, layout=(g)-> (exp10(spectral_layout(g)[1]), log(float(vertices(g)))),nodelabel=1:nv(g), nodelabelsize=1, nodefillc=nodefillc)
end

function Psis(ϵ::Real)
    return [1-ϵ ϵ;
            ϵ 1-ϵ]
end