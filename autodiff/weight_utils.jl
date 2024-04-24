# todo delete?
function create_kernel(n_input::Int64, n_output::Int64; kernel_size = 3)
    stddev = sqrt(1 / (n_input * 9))
    return stddev .- rand(kernel_size, kernel_size, n_input, n_output) * stddev * 2
end

# todo delete?
function initialize_uniform_bias(in_features::Int64, out_features::Int64)
    k = sqrt(1 / in_features)
    return k .- 2 * rand(out_features) * k
end

# todo rename second hidden layer name
function init_weights()
    kernel_weight = Variable(create_kernel(1, 6))

    l1_hidden_weight = Variable(randn(84, 13*13*6), name = "wh")
    l1_hidden_bias = Variable(initialize_uniform_bias(13*13*6, 84))
    
    l2_hidden_weight = Variable(randn(10, 84), name = "wo");
    l2_hidden_bias = Variable(initialize_uniform_bias(84, 10));

    return kernel_weight, l1_hidden_weight, l2_hidden_weight
end