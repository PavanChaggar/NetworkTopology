include("../Models/Models.jl")
gr()

N = 83

# Make a simple weighted random graph and graph Laplacian
GW = MakeSimpleWeightedGraph(N, 0.8)
L = laplacian_matrix(GW)

heatmap(Array(L))


# Set up ODE problem
p = [0.1, 2.0, 1.0]

protein = zeros(N)
protein[rand(1:N)] = 0.1

u0 = [protein; zeros(N)]
t_span = (0.0,10.0)
prob = ODEProblem(NetworkAtrophy, u0, t_span, p)

sol = solve(prob, Tsit5(), saveat=0.1)

# make data and plot against solution
#data = clamp.(Array(sol) + 0.02 * randn(size(Array(sol))), 0.0,1.0)
data = Array(sol)

atrophy_data = data[N+1:end,:]

plot(sol, vars=(1:N))
scatter!(0:0.1:10,data[1:N,:]')

plot(sol, vars=(N+1:2N))
scatter!(0:0.1:10,data[N+1:2N,:]')

# Sample from Prior and get prior predictive models 
prior_chain = sample(NetworkAtrophy(data,prob), NUTS(0.65), Prior(), 10_000)

@time sample(NetworkAtrophy(data,prob), Prior(), 1_000)

prior_chain_array = Array(prior_chain)


plot_predictive(chain_array, prob, sol, data, 2)

model = NetworkAtrophyOnly(atrophy_data,prob)
chain = sample(model, NUTS(0.65), 1_000)

chain_array = Array(chain)

plot_predictive(chain_array, prob, sol, data, 5)

plot(ylims=(0,1))
for k in 1:100
    par = chain_array[rand(1:1_000), 1:14]
    resol = solve(remake(prob,u0=par[4:13], p=[par[3],par[1],par[2]]),Tsit5(),saveat=0.1)
    plot!(Array(resol)[1:5,:]', alpha=0.1, color = "#BBBBBB", legend = false)
end
scatter!(data[1:5,:]', legend = false)
plot!(Array(sol[1:5,:])', w=2, legend = false)