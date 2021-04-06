using DifferentialEquations, LightGraphs, Plots, SimpleWeightedGraphs, LinearAlgebra, Turing, Base.Threads, MCMCChains
Turing.setadbackend(:forwarddiff)

function NetworkCoupledFKPP(du, u0, p, t; L=L)
    n = Int(length(u0)/2)

    x = u0[1:n]
    y = u0[n+1:2n]

    ρ, α, β = p

    du[1:n] .= -ρ * L * x .+ α .* x .* (1.0 .- x)
    du[n+1:2n] .= β * x .* (1.0 .- y)
end

@model function NetworkAtrophyPM(data, problem)
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

@model function NetworkAtrophy(data, problem, ::Type{T} = Float64) where {T} 
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