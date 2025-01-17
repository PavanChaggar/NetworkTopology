using Turing

@model function linear_reg(x, y, σ = 0.1)
    β ~ Normal(0, 1)

    for i ∈ eachindex(y)
        y[i] ~ Normal(β * x[i], σ)
    end
end

@model function linear_reg_vec(x, y, σ = 0.1)
    β ~ Normal(0, 1)
    y ~ MvNormal(β .* x, σ)
end

f(x) = 2 * x + 0.1 * randn()

Δ = 0.1
xs_train = 0:Δ:10; 
ys_train = f.(xs_train);

xs_test = [10 + Δ, 10 + 2 * Δ]; 
ys_test = f.(xs_test);

# Infer
m_lin_reg = linear_reg(xs_train, ys_train);
chain_lin_reg = sample(m_lin_reg, NUTS(100, 0.65), 200);

# Predict on two last indices
m_lin_reg_test = linear_reg(xs_test, fill(missing, length(ys_test)));    
predictions = Turing.Inference.predict(m_lin_reg_test, chain_lin_reg)

ys_pred = vec(mean(Array(group(predictions, :y)); dims = 1))