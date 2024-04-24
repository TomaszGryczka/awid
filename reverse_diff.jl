using Flux

include("autoDiff.jl")
using .AutoDiff

# function custom_gradient(net, x, y)
#     # Define the loss function
#     loss(model) = Flux.logitcrossentropy(model(x), y)
    
#     # Compute gradients of the loss with respect to the model parameters
#     grads = Flux.gradient(loss, net)
    
#     return grads
# end



x = Variable(5.0, 0.0)
y = sin(x*x)

rosenbrock(x, y) = (Constant(1.0) .- x .* x) .+ Constant(100.0) .* (y .- x .* x) .* (y .- x .* x)
x = Variable([0.])
y = Variable([0.])
graph = topological_sort(rosenbrock(x, y))

v = -1:.1:+1
n = length(v)
z = zeros(n, n)
dzdx = zeros(n, n)
dzdy = zeros(n, n)

for i = 1:n, j = 1:n
    x.output .= v[i]
    y.output .= v[j]

    z[i, j] = first(forward!(graph))
    backward!(graph)

    dzdx[i, j] = first(gradient(x))
    dzdy[i, j] = first(gradient(y))
end