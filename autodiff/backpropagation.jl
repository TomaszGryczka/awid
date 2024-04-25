update!(node::Constant, gradient) = nothing
update!(node::GraphNode, gradient) = let
    node.gradient = gradient
    if typeof(node) == Variable
        if isnothing(node.batch_gradient)
            node.batch_gradient = gradient
        else
            node.batch_gradient += gradient
        end
    end
    
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient)
    end
    return nothing
end