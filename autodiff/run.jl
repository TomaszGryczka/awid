include("cnn.jl")

using MLDatasets: MNIST
using Flux

train_data = MNIST(:train)

x = reshape(train_data.features, 28, 28, :)
y = Flux.onehotbatch(train_data.targets, 0:9)

# reduce number of samples for testing
x = x[:, :, 1:5000]
y = y[:, 1:5000]

train(x, y', 3, 100, 1e-2)