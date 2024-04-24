using Test

include("../graph_builder.jl")

x = Variable(5.0, name="x")
two = Constant(2.0)
squared = x^two
sine = sin(squared)

order = topological_sort(sine)

y = forward!(order)
backward!(order)
x.gradient

@info "Derivative: $(x.gradient)"

@test y == -0.13235175009777303
@test x.gradient == 9.912028118634735