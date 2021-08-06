using Turing, DifferentialEquations
using DelimitedFiles, LightGraphs, SimpleWeightedGraphs
using LinearAlgebra
using Base.Threads

subjectid = readdlm("/Users/pavanchaggar/Projects/Connectomes/all_subjects") .|> Int |> vec

conndir = "/Users/pavanchaggar/Projects/Connectomes/standard_connectome/scale1/subjects/"

function get_graph(subjectid, conndir, n)
    M = Array{Float64}(undef, 83 * 83, n)

    for i in 1:n
        ndir = conndir * string(subjectid[i]) * "/fdt_network_matrix"
        ldir = conndir * string(subjectid[i]) * "/fdt_network_matrix_lengths"
        
        na = readdlm(ndir)
        nl = readdlm(ldir)

        A = na ./ nl.^2
        [A[i,i] = 0.0 for i in 1:83]

        M[:,i] .= vec(A) ./ maximum(A)

    end
    M
end

As = get_graph(subjectid, conndir, 2)

diagI = vec(Diagonal(Int.(ones(83))))
index = findall(x->x==0, diagI)

@model function fit(A)
    e ~ filldist(truncated(Normal(),0,1),83*83 - 83)
    σ ~ filldist(InverseGamma(2, 3), 83*83 - 83)
    
    for i in 1:2
        A[index,i] ~ MvNormal(e, σ)
    end
end

model = fit(As)
model()

advi = ADVI(10, 1000)
vi(model, advi)

As[index,1]