include("topological_sorter.jl")
include("conv_operation.jl")
include("weight_utils.jl")
include("backward.jl")
include("forward.jl")

using LinearAlgebra
using Statistics

cross_entropy_loss(y_hat::GraphNode, y::GraphNode) = BroadcastedOperator(cross_entropy_loss, y_hat, y)
forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y) =
    let
		global num_of_clasiffications
        global num_of_correct_clasiffications
        num_of_clasiffications += 1
        if argmax(y_hat) == argmax(y)
            num_of_correct_clasiffications += 1
        end
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

function build_graph(x, y, kernel_weights, hidden_weights, output_weights)
	l1 = convolution(x, kernel_weights) |> relu
	l2 = maxpool2d(l1)
	l3 = flatten(l2)
	l4 = dense(l3, hidden_weights) |> relu
	l5 = dense(l4, output_weights)

	e = cross_entropy_loss(l5, y)

	return topological_sort(e)
end

function update_weights!(graph::Vector, lr::Float64)
    for node in graph
        if isa(node, Variable) && hasproperty(node, :gradient)
            node.output -= lr * node.gradient
            node.gradient .= 0
        end
    end
end

num_of_correct_clasiffications = 0
num_of_clasiffications = 0

function train(x, y, epochs)
	kernel_weights, hidden_weights, output_weights = init_weights()

	for i=1:epochs
		
		epoch_loss = 0.0
		num_of_samples = size(x, 3)

		global num_of_correct_clasiffications = 0
		global num_of_clasiffications = 0
		
		for j=1:num_of_samples
			train_x = Constant(x[:, :, j])
			train_y = Constant(y[j, :])
			graph = build_graph(train_x, train_y, kernel_weights, hidden_weights, output_weights)
	
			epoch_loss += (forward!(graph) / num_of_samples)
			backward!(graph)
			update_weights!(graph, 1e-4)
		end
		
		println("Epoka: ", i,". Strata: ", epoch_loss)
		println("Dokładność: ", num_of_correct_clasiffications/num_of_clasiffications, " (rozpoznano ", num_of_correct_clasiffications, "/", num_of_clasiffications, ")\n")
	end
end
