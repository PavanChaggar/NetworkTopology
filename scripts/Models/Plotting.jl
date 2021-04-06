using Plots, MCMCChains

function plot_predictive(chain_array, prob, sol, data, node::Int)
    plot(Array(sol)[node,:], w=2, legend = false)
    for k in 1:300
        par = chain_array[rand(1:1_000), 1:13]
        resol = solve(remake(prob,u0=par[4:13], p=[par[3],par[1],par[2]]),Tsit5(),saveat=0.1)
        plot!(Array(resol)[node,:], alpha=0.5, color = "#BBBBBB", legend = false)
    end
    scatter!(data[node,:], legend = false)
end