using DifferentialEquations, LightGraphs, Plots, SimpleWeightedGraphs, LinearAlgebra, Turing, Base.Threads, MCMCChains
Turing.setadbackend(:forwarddiff)

"""
    MakeSimpleWeightedGraph(n, p)

Create a simple weighted random graph of `n` nodes with connection probability `p`.
"""
function MakeSimpleWeightedGraph(n::Int64, p::Float64)
    GW = SimpleWeightedGraph(erdos_renyi(n, p))
    for e in edges(GW)
        add_edge!(GW, src(e), dst(e), rand())
    end
    return GW
end

function NetworkAtrophy(du, u0, p, t; L=L)
    n = Int(length(u0)/2)

    x = u0[1:n]
    y = u0[n+1:2n]

    ρ, α, β = p

    du[1:n] .= -ρ * L * x .+ α .* x .* (1.0 .- x)
    du[n+1:2n] .= β * x .* (1.0 .- y)
end

@model function NetworkAtrophy(data, problem)
    σ ~ InverseGamma(2, 3)
    r ~ truncated(Normal(0, 1), 0, Inf)
    a ~ truncated(Normal(0, 2), 0, Inf)
    b ~ truncated(Normal(0, 2), 0, Inf)

    u ~ filldist(truncated(Normal(0, 0.1), 0, 1), 166)

    p = [r, a, b]

    prob = remake(problem, u0=u, p=p)
    
    predicted = solve(prob, Tsit5(), saveat=0.1)
    @threads for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end

    return r, a, b, u
end



function plot_predictive(chain_array, prob, sol, data, node::Int)
    plot(Array(sol)[node,:], w=2, legend = false)
    for k in 1:300
        par = chain_array[rand(1:1_000), 1:13]
        resol = solve(remake(prob,u0=par[4:13], p=[par[3],par[1],par[2]]),Tsit5(),saveat=0.1)
        plot!(Array(resol)[node,:], alpha=0.5, color = "#BBBBBB", legend = false)
    end
    scatter!(data[node,:], legend = false)
end

@model function NetworkAtrophyOnly(data, problem, ::Type{T} = Float64) where {T} 
    n = Int(size(data)[1])

    σ ~ InverseGamma(2, 3)
    r ~ truncated(Normal(0, 1), 0, Inf)
    a ~ truncated(Normal(0, 2), 0, Inf)
    b ~ truncated(Normal(0, 2), 0, Inf)

    u ~ filldist(truncated(Normal(0, 0.1), 0, 1), 2n)

    p = [r, a, b]

    prob = remake(problem, u0=u, p=p)
    
    predicted = solve(prob, Tsit5(), saveat=0.1)
    @threads for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[n+1:end,i], σ)
    end

    return r, a, b, u
end

read_subjects(csv_path::String) = Int.(readdlm(csv_path))
	
symmetrise(M) = 0.5 * (M + transpose(M))

max_norm(M) = M ./ maximum(M)

adjacency_matrix(file::String) = sparse(readdlm(file))

laplacian_matrix(A::Array{Float64,2}) = SimpleWeightedGraphs.laplacian_matrix(SimpleWeightedGraph(A))
    
function load_connectome(subjects, subject_dir, N, size, length)
    
    M = Array{Float64}(undef, size, size, N)
    
    if length == true
        connectome_type = "/fdt_network_matrix_lengths"
    else
        connectome_type = "/fdt_network_matrix"
    end
    
    for i ∈ 1:N
        file = subject_dir * string(subjects[i]) * connectome_type
        M[:,:,i] = symmetrise(adjacency_matrix(file))
    end
    
    return M
end

mean_connectome(M) = mean(M, dims=3)[:,:]

function diffusive_weight(An, Al)
    A = An ./ Al.^2
    [A[i,i] = 0.0 for i in 1:size(A)[1]]
    return A
end	

function plot_predictive(chain_array, prob, sol, data, node::Int)
    plot(Array(sol)[node,:], w=2, legend = false)
    for k in 1:300
        par = chain_array[rand(1:1_000), 1:23]
        resol = solve(remake(prob,u0=par[4:23], p=[par[3],par[1],par[2]]),Tsit5(),saveat=0.1)
        plot!(Array(resol)[node,:], alpha=0.5, color = "#BBBBBB", legend = false)
    end
    scatter!(data[node,:], legend = false)
end