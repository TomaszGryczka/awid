function init_weights()
    kernel_weights = Variable(kernel_initialization(3, 1, 6))
    hidden_weights = Variable(he_initialization(13*13*6, 84), name = "wh")
    output_weights = Variable(randn(10, 84), name = "wo");

    return kernel_weights, hidden_weights, output_weights
end

function he_initialization(n_in, n_out)
    return randn(n_out, n_in) * sqrt(2/n_in)
end

function kernel_initialization(kernel_size, n_in, n_out)
    return randn(kernel_size, kernel_size, n_in, n_out) * sqrt(2/n_in)
end