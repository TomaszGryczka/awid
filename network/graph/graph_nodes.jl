import Base: show, summary

abstract type GraphNode end
abstract type Operator <: GraphNode end

mutable struct Constant <: GraphNode
    output :: Any
end

mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
    name::String
    batch_gradient::Any
    Variable(output; name = "?") = new(output, nothing, name, nothing)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    cache::Any
    function BroadcastedOperator(fun, inputs...)
       return new{typeof(fun)}(inputs, nothing, nothing, nothing) 
    end
end