import LinearAlgebra: diagm

dense(w, n, m, v, f) = f.(reshape(w, n, m) * v)
mean_squared_loss(y, ŷ) = sum(0.5(y - ŷ).^2)
sigmoid(x) = one(x) / (one(x) + exp(-x))
linear(x) = x
eye(n) = diagm(ones(n))
diagonal(v) = diagm(v)

function ∇W(x, x̂, ŷ, y, Wo)
    @info "x", x
    @info "x̂", x̂
    @info "y", y
    @info "Wo", Wo
    @info "ŷ", ŷ

    # mean_squared_loss
    Eŷ = ŷ - y
    @info "strata", Eŷ

    # liniowa funkcja aktywacji
    ŷȳ = ŷ |> length |> eye
    @info "ŷȳ", ŷȳ

    # sumowanie (W*x)
    ȳWo = x̂ |> transpose
    @info "ȳWo", ȳWo

    ȳx̂ = Wo |> transpose
    @info "ȳx̂", ȳx̂

    # sigmoidalna f. aktywacji
    x̂x̄ = x̂ .* (1.0 .- x̂) |> diagonal
    # sumowanie (W*x) wzg. wag
    x̄Wh = x |> transpose
    # reguła łańcuchowa
    Eȳ = ŷȳ * Eŷ
    Ex̂ = ȳx̂ * Eȳ
    Ex̄ = x̂x̄ * Ex̂
    EWo = Eȳ * ȳWo
    EWh = Ex̄ * x̄Wh
    return EWo, EWh
end

function net(x, wh, wo, y)
    x̂ = dense(wh, 10, 2, x, sigmoid)
    ŷ = dense(wo, 1, 10, x̂, linear)
    EWo, EWh = ∇W(x, x̂, ŷ, y, wo)
    dWo .= EWo
    dWh .= EWh
    E = mean_squared_loss(y, ŷ)
end

Wh = randn(10,2)
Wo = randn(1,10)
x, y = [1.98; 4.434], [0.064]

dWo = similar(Wo)
dWh = similar(Wh)

epochs = 1
for i=1:epochs
    global Wh, Wo
    E = net(x, Wh, Wo, y)
    Wh -= 0.1dWh
    Wo -= 0.1dWo
    @info "E", E
end


