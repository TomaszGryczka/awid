include("topological_sorter.jl")
include("conv_operation.jl")
include("weight_utils.jl")

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


# too l2 is not second layer
function build_graph(x, y, kernel_weight, l1_hidden_weight, l1_hidden_bias, l2_hidden_weight, l2_hidden_bias)
	con = convolution(x, kernel_weight) |> relu |> maxpool2d |> flatten
	l1 = dense(con, l1_hidden_weight, l1_hidden_bias) |> relu
	l2 = dense(l1, l2_hidden_weight, l2_hidden_bias)
	e = cross_entropy_loss(l2, y)
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
	kernel_weight, l1_hidden_weight, l1_hidden_bias, l2_hidden_weight, l2_hidden_bias = init_weights()

	for i=1:epochs
		
		epoch_loss = 0.0
		global num_of_correct_clasiffications = 0
		global num_of_clasiffications = 0
		
		for j=1:size(x, 3)
			train_x = Constant(x[:, :, j])
			train_y = Constant(y[j, :])
			graph = build_graph(train_x, train_y, kernel_weight, l1_hidden_weight, l1_hidden_bias, l2_hidden_weight, l2_hidden_bias)
	
			epoch_loss += forward!(graph) / size(x, 3)
			backward!(graph)
			update_weights!(graph, 1e-4)
		end
		println("Epoka: ", i,". Strata: ", epoch_loss)
		println("Dokładność: ", num_of_correct_clasiffications/num_of_clasiffications, " (rozpoznano ", num_of_correct_clasiffications, "/", num_of_clasiffications, ")\n")
	end
end
