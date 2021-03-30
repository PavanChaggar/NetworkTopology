include("../Models/Models.jl")
gr()

N = 83 

# Make a simple weighted random graph and graph Laplacian
GW = MakeSimpleWeightedGraph(N,0.1)
L = laplacian_matrix(GW)

# Set up ODE problem
p = [0.1, 2.0, 1.0]

protein = zeros(83)
protein[rand(1:83)] = 0.1

u0 = [protein; zeros(83)]
t_span = (0.0,10.0)
prob = ODEProblem(NetworkAtrophy, u0, t_span, p)

sol = solve(prob, Tsit5(), saveat=0.1)

# make data and plot against solution
data = clamp.(Array(sol) + 0.02 * randn(size(Array(sol))), 0.0,1.0)

plot(sol, vars=(1:83))
scatter!(0:0.1:10,data[1:83,:]')

plot(sol, vars=(84:166))
scatter!(0:0.1:10,data[84:166,:]')

# Sample from Prior and get prior predictive models 
prior_chain = sample(NetworkAtrophy(data,prob), Prior(), 10_000)

chain_array = Array(prior_chain)


plot_predictive(chain_array, prob, sol, data, 2)

model = NetworkAtrophy(data,prob)
chain = sample(model, NUTS(0.65), 1_000)
