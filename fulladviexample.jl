using RDatasets
using Turing
using Turing: Variational
using LinearAlgebra
using Bijectors
using Bijectors: Scale, Shift
using Plots 
using StatsPlots

# Hide the progress prompt while sampling.
Turing.turnprogress(true);

data = RDatasets.dataset("datasets", "mtcars");

# Show the first six rows of the dataset.
first(data, 6)

# Function to split samples.
function split_data(df, at = 0.70)
    r = size(df,1)
    index = Int(round(r * at))
    train = df[1:index, :]
    test  = df[(index+1):end, :]
    return train, test
end

# A handy helper function to rescale our dataset.
function standardize(x)
    return (x .- mean(x, dims=1)) ./ std(x, dims=1), x
end

# Another helper function to unstandardize our datasets.
function unstandardize(x, orig)
    return (x .+ mean(orig, dims=1)) .* std(orig, dims=1)
end

# Remove the model column.
select!(data, Not(:Model))

# Standardize our dataset.
(std_data, data_arr) = standardize(Matrix(data))

# Split our dataset 70%/30% into training/test sets.
train, test = split_data(std_data, 0.7)

# Save dataframe versions of our dataset.
train_cut = DataFrame(train, names(data))
test_cut = DataFrame(test, names(data))

# Create our labels. These are the values we are trying to predict.
train_label = train_cut[:, :MPG]
test_label = test_cut[:, :MPG]

# Get the list of columns to keep.
remove_names = filter(x->!in(x, [:MPG, :Model]), names(data))

# Filter the test and train sets.
train = Matrix(train_cut[:,remove_names]);
test = Matrix(test_cut[:,remove_names]);

@model linear_regression(x, y, n_obs, n_vars, ::Type{T}=Vector{Float64}) where {T} = begin
    # Set variance prior.
    σ₂ ~ truncated(Normal(0,100), 0, Inf)
    
    # Set intercept prior.
    intercept ~ Normal(0, 3)
    
    # Set the priors on our coefficients.
    coefficients ~ MvNormal(zeros(n_vars), 10 * ones(n_vars))
    
    # Calculate all the mu terms.
    mu = intercept .+ x * coefficients
    y ~ MvNormal(mu, σ₂)
end;

n_obs, n_vars = size(train)
m = linear_regression(train, train_label, n_obs, n_vars);

q0 = Variational.meanfield(m)
typeof(q0)

advi = ADVI(10, 1_000)

q = vi(m, advi, q0)

d = length(q)

base_dist = Turing.DistributionsAD.TuringDiagMvNormal(zeros(d), ones(d))

to_constrained = inv(bijector(m));

function getq(θ)
    d = length(θ) ÷ 2
    A = @inbounds θ[1:d]
    b = @inbounds θ[d + 1: 2 * d]
    
    b = to_constrained ∘ Shift(b; dim = Val(1)) ∘ Scale(exp.(A); dim = Val(1))
    
    return transformed(base_dist, b)
end

q_mf_normal = vi(m, advi, getq, randn(2 * d));

d = 13

function getq(θ)
    #length \theta = d^2 + d 
    offset = 0
    L = LowerTriangular(reshape(@inbounds(θ[offset + 1: offset + d^2]), (d, d)))
    offset += d^2
    b = @inbounds θ[offset + 1: offset + d]
    
    # For this to represent a covariance matrix we need to ensure that the diagonal is positive.
    # We can enforce this by zeroing out the diagonal and then adding back the diagonal exponentiated.
    D = Diagonal(diag(L))
    A = L - D + exp(D) # exp for Diagonal is the same as exponentiating only the diagonal entries
    
    b = to_constrained ∘ Shift(b; dim = Val(1)) ∘ Scale(A; dim = Val(1))
    
    return transformed(base_dist, b)
end

advi = ADVI(10, 10)

q_full_normal = vi(m, advi, getq, randn(d^2 + d); optimizer = Variational.DecayedADAGrad(1e-2));

A = q_full_normal.transform.ts[1].a

heatmap(cov(A * A'))

θ =randn(d^2 + d) 
offset = 0
L = LowerTriangular(reshape(@inbounds(θ[offset + 1: offset + d^2]), (d, d)))
