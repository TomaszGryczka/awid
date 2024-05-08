include("./graph/topological_sorter.jl")
include("./training/conv_operations.jl")
include("./training/weight_utils.jl")
include("./training/backpropagation.jl")
include("./training/forward.jl")
include("./training/cross_entropy_loss.jl")
include("./graph/graph.jl")

num_of_correct_clasiffications = 0
num_of_clasiffications = 0

function train(x, y, epochs, batch_size, learining_rate)
	kernel_weights, hidden_weights, output_weights = init_weights()

	@time for i=1:epochs
		
		epoch_loss = 0.0
		num_of_samples = size(x, 3)

		global num_of_correct_clasiffications = 0
		global num_of_clasiffications = 0

		train_x = Constant(x[:, :, 1])
		train_y = Constant(y[1, :])

		graph = build_graph(train_x, train_y, kernel_weights, hidden_weights, output_weights)
		
		for j=2:num_of_samples	
			epoch_loss += forward!(graph)
			backward!(graph)
			
			if j % batch_size == 0
				update_weights!(graph, learining_rate, batch_size)
			end

			train_x.output = x[:, :, j]
			train_y.output = y[j, :]
		end
		
		println("Epoch: ", i,". Average epoch loss: ", epoch_loss  / num_of_samples)
		println("Train accuracy: ", num_of_correct_clasiffications/num_of_clasiffications, " (recognized ", num_of_correct_clasiffications, "/", num_of_clasiffications, ")\n")
	end

	return kernel_weights, hidden_weights, output_weights
end


function test(x, y, kernel_weights, hidden_weights, output_weights)
	num_of_samples = size(x, 3)

	global num_of_correct_clasiffications = 0
	global num_of_clasiffications = 0

	train_x = Constant(x[:, :, 1])
	train_y = Constant(y[1, :])

	graph = build_graph(train_x, train_y, kernel_weights, hidden_weights, output_weights)

	for j=2:num_of_samples
		forward!(graph)

		train_x.output = x[:, :, j]
		train_y.output = y[j, :]
	end

	println("\n")
	println("Test accuracy: ", num_of_correct_clasiffications/num_of_clasiffications, " (recognized ", num_of_correct_clasiffications, "/", num_of_clasiffications, ")\n")
end