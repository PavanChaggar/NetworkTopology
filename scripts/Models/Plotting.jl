using Plots, MCMCChains
using DelimitedFiles

function plot_predictive(chain_array, prob, sol, data, node::Int)
    plot(Array(sol)[node,:], w=2, legend = false)
    for k in 1:300
        par = chain_array[rand(1:1_000), 1:13]
        resol = solve(remake(prob,u0=par[4:13], p=[par[3],par[1],par[2]]),Tsit5(),saveat=0.1)
        plot!(Array(resol)[node,:], alpha=0.5, color = "#BBBBBB", legend = false)
    end
    scatter!(data[node,:], legend = false)
end

function plot_connectome(coord_path, size, alpha)
    coords = readdlm(coord_path)
    x, y, z = coords[:,1], coords[:,2], coords[:,3]
    scatter(x, y, z, grid=false, showaxis=false, markersize=size, markeralpha=alpha)
end

function plot_connectome(coord_path, A, size, alpha)
    fig = plot_connectome(coord_path, size, alpha)
    coordindex = findall(x->x>0, A)
    for i ∈ 1:length(coordindex)
        j, k = coordindex[i][1], coordindex[i][2]
        plot!(fig, x[[j,k]], y[[j,k]], z[[j,k]], primary=false)
    end
    display(fig)
end