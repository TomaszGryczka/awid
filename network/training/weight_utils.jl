function init_weights()
    kernel_weights = Variable(randn(3, 3, 1, 6))
    hidden_weights = Variable(randn(84, 13*13*6), name = "wh")
    output_weights = Variable(randn(10, 84), name = "wo");

    return kernel_weights, hidden_weights, output_weights
end