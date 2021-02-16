using DifferentialEquations, LightGraphs, Random, Turing
using Turing: Variational
using LinearAlgebra
using Bijectors
using Bijectors: Scale, Shift
using Plots
using StatsPlots
using ReverseDiff
using DiffEqSensitivity
using Zygote

function lotka_volterra(du,u,p,t)
    x, y = u
    α, β, γ, δ  = p
    du[1] = (α - β*y)x # dx =
    du[2] = (δ*x - γ)y # dy = 
  end
 
@model function fitlv(data, prob)
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5,0.5),0.5,2.5)
    β ~ truncated(Normal(1.2,0.5),0,2)
    γ ~ truncated(Normal(3.0,0.5),1,4)
    δ ~ truncated(Normal(1.0,0.5),0,2)
    p = [α,β,γ,δ]    
    prob = remake(prob, p=p)

    predicted = solve(prob,saveat=0.1)
    for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end
end;

p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0,1.0]
prob1 = ODEProblem(lotka_volterra,u0,(0.0,10.0),p)
sol1 = solve(prob1,Tsit5(),saveat=0.1)
odedata = Array(sol1) #+ 0.8 * randn(size(Array(sol1)))

Turing.setadbackend(:forwarddiff)
model = fitlv(odedata, prob1)
chain = sample(model, NUTS(.65),1000)