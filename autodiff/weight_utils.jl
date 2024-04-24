# todo delete?
function create_kernel(n_input::Int64, n_output::Int64; kernel_size = 3)
    stddev = sqrt(1 / (n_input * 9))
    return stddev .- rand(kernel_size, kernel_size, n_input, n_output) * stddev * 2
end

function init_weights()
    kernel_weights = Variable(create_kernel(1, 6))
    hidden_weights = Variable(randn(84, 13*13*6), name = "wh")
    output_weights = Variable(randn(10, 84), name = "wo");

    return kernel_weights, hidden_weights, output_weights
end