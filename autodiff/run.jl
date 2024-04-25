include("cnn.jl")

using MLDatasets: MNIST
using Flux

train_data = MNIST(:train)
test_data = MNIST(:test)

x_train = reshape(train_data.features, 28, 28, :)
y_train = Flux.onehotbatch(train_data.targets, 0:9)
x_test = reshape(test_data.features, 28, 28, :)
y_test = Flux.onehotbatch(test_data.targets, 0:9)

# reduce number of samples for testing
x_train = x_train[:, :, 1:5000]
y_train = y_train[:, 1:5000]
x_test = x_test[:, :, 1:5000]
y_test = y_test[:, 1:5000]

# training
kernel_weights, hidden_weights, output_weights =  train(x_train, y_train', 3, 100, 1e-2)

# testing
test(x_test, y_test', kernel_weights, hidden_weights, output_weights)