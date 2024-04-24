include("graph_builder.jl")
include("conv_operation.jl")
# using Flux
using LinearAlgebra
using Statistics
# using MLDatasets: MNIST
using Random
Random.seed!(1)

function conv(w, b, x, activation)
	out = conv(x, w) .+ b
	return activation(out)
end

cross_entropy_loss(y_hat::GraphNode, y::GraphNode) = BroadcastedOperator(cross_entropy_loss, y_hat, y)
forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y) =
    let
        y_hat = y_hat .- maximum(y_hat)
        y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
        loss = sum(log.(y_hat) .* y) * -1.0
        return loss
    end
backward(node::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y, g) =
    let
        y_hat = y_hat .- maximum(y_hat)
        y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
        return tuple(g .* (y_hat - y))
    end




dense(w, x, activation) = activation(w * x)
mean_squared_loss(y, ŷ) = Constant{Float64}(0.5) .* (y .- ŷ) .^ Constant{Float64}(2)
flatten(x) = flatten(x)

update_weight!(node, learning_rate) = node.output -= learning_rate .* node.gradient

is_weight(node) = occursin("w", node.name)

is_bias(node) = occursin("b", node.name)

is_parameter(node) = is_weight(node) || is_bias(node)

is_x(node) = occursin("x", node.name)

is_y(node) = occursin("y", node.name)

has_name(node) = hasproperty(node, :name)

average_bias_gradient!(node) = node.gradient = mean(node.gradient, dims = (1, 2))

function save_param_gradients!(graph, gradients_in_batch)
	for (idx, node) in enumerate(graph)
		if has_name(node) && is_parameter(node)
			if is_bias(node)
				average_bias_gradient!(node)
			end
			if !haskey(gradients_in_batch, node.name)
				gradients_in_batch[node.name] = Vector{AbstractArray{<:Real, 4}}()
			end
			push!(gradients_in_batch[node.name], node.gradient)
			# println("save_param_gradients!: ", typeof(node.gradient))
		end
	end
end

function learning_iteration!(graph, learning_rate, if_print)
	forward!(graph)
	backward!(graph)
	if if_print
		for (i, n) in enumerate(graph)
			if typeof(n) <: Variable
				println("Node $i")
				println(n.name)
				println(n.output)
				println(size(n.output))
				println(n.gradient)
				println()
			end
		end
	end
end

input_size = 28
kernel_size = 3
input_channels = 1
out_channels = 4
x_global = Variable(randn(input_size, input_size, input_channels, 1)::Array{Float64, 4}, name = "x")
# wh_global = Variable(randn(kernel_size, kernel_size, input_channels, out_channels)::Array{Float64, 4}, name = "wh")
# bh_global = Variable(randn(1, 1, out_channels, 1)::Array{Float64, 4}, name = "bh")
wo_global = Variable(randn((784, 1))::Matrix{Float64}, name = "wo")
y_global = Variable(randn(10, 1), name = "y")


weight_hidden = Variable(randn(10, 2), name = "wh")
weight_output = Variable(randn(1, 10), name = "wo")

function model(x, wh, wo, y)
	x̂ = dense(wh, x, linear)
	ŷ = dense(wo, x̂, linear)
	# println(ŷ)
	e = mean_squared_loss(ŷ, y)
	# println(e)
	return topological_sort(e)
	# return topological_sort(ŷ)
end


function build_graph()
	# x̂ = conv(wh_global, bh_global, x_global, relu)
	# x̂.name = "x̂"
	# x̂ = flatten(x̂)
	x̂ = flatten(x_global)
	x̂.name = "x̂"
	ŷ = dense(wo_global, x̂, relu)
	ŷ.name = "ŷ"
	e = cross_entropy_loss(y_global, ŷ)
	e.name = "loss"
	return topological_sort(e), x_global, y_global
end

# Define a function to train a neural network model
function train_model(x, y, learning_rate, n_iterations, if_print)
	# Build the computation graph and initialize the input nodes
	graph, x_node, y_node = build_graph()
	# Create a vector to store the average loss after each iteration
	avg_losses = Vector{Float64}()
	# Create a vector to store times of forward and backward pass
	times = Vector{Float64}()

	# Iterate over the specified number of batches
	for iter ∈ 1:n_iterations
		# Create empty vectors to store the losses and gradients in the current batch
		losses_in_batch = Vector{Float64}()
		gradients_in_batch = Dict{String, Vector{AbstractArray}}()
		# Create a dictionary to store the mean gradients across the batch
		mean_gradients = Dict{String, AbstractArray}()
		# Print the batch number
		println("Batch $iter")

		# Iterate over each input in the batch
		for i ∈ 1:size(x, 4)
			# Get the current input and reshape it to match the input node shape
			curr_x = x[:, :, :, i]
			curr_x = reshape(curr_x, size(curr_x, 1), size(curr_x, 2), size(curr_x, 3), 1)
			# Get the current target output
			curr_y = y[i, :]
			# Set the input and output nodes to the current values
			x_node.output = curr_x
			y_node.output = curr_y
			# Perform a forward pass through the graph and update the gradients
			start = time()
			learning_iteration!(graph, learning_rate, if_print)
			finish = time()
			# Save the time of the forward and backward pass
			push!(times, finish - start)
			# Save the gradients for each parameter in the graph
			save_param_gradients!(graph, gradients_in_batch)
			# Get the loss for the current input and append it to the losses in the batch vector
			loss = graph[end].output[1]
			push!(losses_in_batch, loss)
		end

		# Compute the mean gradients for each parameter in the graph
		for (key, value) in gradients_in_batch
			matrixes = gradients_in_batch[key]
			sum = matrixes[1]
			n_matrixes = length(matrixes)
			for i ∈ 2:n_matrixes
				sum .+= matrixes[i]
			end
			mean = sum ./ n_matrixes
			mean_gradients[key] = mean
		end

		# Update each parameter in the graph using the mean gradients
		for (idx, node) in enumerate(graph)
			if has_name(node) && is_parameter(node)
				node.output -= learning_rate .* mean_gradients[node.name]
			end
		end

		# Compute the average loss for the batch and append it to the average losses vector
		avg_loss = mean(losses_in_batch)
		push!(avg_losses, avg_loss)
		
	end
	println(y_node)
	# Return the vector of average losses
	return avg_losses, graph, times
end

function update_weights!(graph::Vector, lr::Float64)
    for node in graph
        if isa(node, Variable) && hasproperty(node, :gradient)
            # node._gradient ./= batch_size
            node.output -= lr * node.gradient
            node.gradient .= 0
        end
    end
end

# test_ds = MNIST(:test)
# train_ds = MNIST(:train)
# x_train = train_ds.features
# y_train = train_ds.targets
# x_train = reshape(x_train, size(x_train, 1), size(x_train, 2), 1, size(x_train, 3))
# y_train = y_train .== 5
# x_train = x_train[:, :, :, 1:1]
# y_train = y_train[1:1, :]
# println(Flux.onehotbatch(y_train, 0:9))

# x = [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.21568628 0.53333336 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.6745098 0.99215686 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.07058824 0.8862745 0.99215686 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.19215687 0.07058824 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.67058825 0.99215686 0.99215686 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.11764706 0.93333334 0.85882354 0.3137255 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.09019608 0.85882354 0.99215686 0.83137256 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.14117648 0.99215686 0.99215686 0.6117647 0.05490196 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.25882354 0.99215686 0.99215686 0.5294118 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.36862746 0.99215686 0.99215686 0.41960785 0.003921569 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.09411765 0.8352941 0.99215686 0.99215686 0.5176471 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.6039216 0.99215686 0.99215686 0.99215686 0.6039216 0.54509807 0.043137256 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.44705883 0.99215686 0.99215686 0.95686275 0.0627451 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.011764706 0.6666667 0.99215686 0.99215686 0.99215686 0.99215686 0.99215686 0.74509805 0.13725491 0.0 0.0 0.0 0.0 0.0 0.15294118 0.8666667 0.99215686 0.99215686 0.52156866 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.07058824 0.99215686 0.99215686 0.99215686 0.8039216 0.3529412 0.74509805 0.99215686 0.94509804 0.31764707 0.0 0.0 0.0 0.0 0.5803922 0.99215686 0.99215686 0.7647059 0.043137256 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.07058824 0.99215686 0.99215686 0.7764706 0.043137256 0.0 0.007843138 0.27450982 0.88235295 0.9411765 0.1764706 0.0 0.0 0.18039216 0.8980392 0.99215686 0.99215686 0.3137255 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.07058824 0.99215686 0.99215686 0.7137255 0.0 0.0 0.0 0.0 0.627451 0.99215686 0.7294118 0.0627451 0.0 0.50980395 0.99215686 0.99215686 0.7764706 0.03529412 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.49411765 0.99215686 0.99215686 0.96862745 0.16862746 0.0 0.0 0.0 0.42352942 0.99215686 0.99215686 0.3647059 0.0 0.7176471 0.99215686 0.99215686 0.31764707 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.53333336 0.99215686 0.9843137 0.94509804 0.6039216 0.0 0.0 0.0 0.003921569 0.46666667 0.99215686 0.9882353 0.9764706 0.99215686 0.99215686 0.7882353 0.007843138 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.6862745 0.88235295 0.3647059 0.0 0.0 0.0 0.0 0.0 0.0 0.09803922 0.5882353 0.99215686 0.99215686 0.99215686 0.98039216 0.30588236 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.101960786 0.6745098 0.32156864 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.105882354 0.73333335 0.9764706 0.8117647 0.7137255 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.6509804 0.99215686 0.32156864 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.2509804 0.007843138 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.9490196 0.21960784 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.96862745 0.7647059 0.15294118 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.49803922 0.2509804 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;;;;]
# y = [0; 1; 0; 0; 0; 0; 0; 0; 0; 0;;;]

x, y = Constant([1.98; 4.434]), Constant([0.064])

# n_batches = 10
# learning_rate = 0.1

# graph, x_node, y_node = build_graph()

# x_node.output = x
# y_node.output = y

epochs = 100

graph = model(x, weight_hidden, weight_output, y)

for i=1:epochs
    global weight_hidden, weight_output

	if i != 1
		global graph = model(x, weight_hidden, weight_output, y)
	end	
    forward!(graph)
	# println(forward!(graph))
	backward!(graph)
	# println(graph)
	update_weights!(graph, 1e-4)
	# println(graph)
end

println(forward!(graph))

# println(forward!(graph))
# backward!(graph)

# start = time()
# losses, graph, times = train_model(x, y, learning_rate, n_batches, false)
# finish = time()
# println("Time of execution: ", finish - start)
# # plot(losses, title = "Loss", xlabel = "Iteration", ylabel = "Loss")

# avg_time = mean(times)
# println("Average time of forward and backward pass: ", avg_time)
# std_time = std(times)
# println("Standard deviation of time of forward and backward pass: ", std_time)
# median_time = median(times)
# println("Median time of forward and backward pass: ", median_time)
# q1 = quantile(times, 0.25)
# q3 = quantile(times, 0.75)
# iqr_time = q3 - q1
# println("Interquartile range of time of forward and backward pass: ", iqr_time)

# test net accuracy
