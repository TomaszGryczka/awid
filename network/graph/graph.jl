function build_graph(x, y, kernel_weights, hidden_weights, output_weights)
	l1 = convolution(x, kernel_weights) |> relu
	l2 = maxpool2d(l1)
	l3 = flatten(l2)
	l4 = dense(l3, hidden_weights) |> relu
	l5 = dense(l4, output_weights) |> identity

	e = cross_entropy_loss(l5, y)

	return topological_sort(e)
end

function update_weights!(graph::Vector, lr::Float64, batch_size::Int64)
    for node in graph
        if isa(node, Variable) && hasproperty(node, :batch_size)
			node.batch_gradient ./= batch_size
            node.output -= lr * node.batch_gradient 
            node.batch_gradient .= 0
        end
    end
end
