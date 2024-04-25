include("topological_sorter.jl")
include("conv_operations.jl")
include("weight_utils.jl")
include("backpropagation.jl")
include("forward.jl")
include("cross_entropy_loss.jl")
include("graph.jl")

num_of_correct_clasiffications = 0
num_of_clasiffications = 0

function train(x, y, epochs, batch_size, learining_rate)
	kernel_weights, hidden_weights, output_weights = init_weights()

	@time for i=1:epochs
		
		epoch_loss = 0.0
		num_of_samples = size(x, 3)

		global num_of_correct_clasiffications = 0
		global num_of_clasiffications = 0
		
		for j=1:num_of_samples
			train_x = Constant(x[:, :, j])
			train_y = Constant(y[j, :])

			graph = build_graph(train_x, train_y, kernel_weights, hidden_weights, output_weights)
	
			epoch_loss += forward!(graph)
			backward!(graph)
			
			if j % batch_size == 0
				update_weights!(graph, learining_rate, batch_size)
			end
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

	for j=1:num_of_samples
		train_x = Constant(x[:, :, j])
		train_y = Constant(y[j, :])

		graph = build_graph(train_x, train_y, kernel_weights, hidden_weights, output_weights)

		forward!(graph)
	end

	println("Test accuracy: ", num_of_correct_clasiffications/num_of_clasiffications, " (recognized ", num_of_correct_clasiffications, "/", num_of_clasiffications, ")\n")
end