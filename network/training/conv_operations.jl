using SparseArrays

import LinearAlgebra: mul!

include("../graph/graph_nodes.jl")

convolution(x::GraphNode, kernel::GraphNode) = BroadcastedOperator(convolution, x, kernel)
forward(::BroadcastedOperator{typeof(convolution)}, x, kernel) =
    let
        x_height, x_width = size(x)
        filter_height, filter_width, _, num_of_kernels = size(kernel)

        output_height = x_height - filter_height + 1
        output_width = x_width - filter_width + 1

        col_x = img2col(x, filter_height, filter_width, output_height, output_width)
        col_kernel = reshape(kernel, :, num_of_kernels)'
        output = zeros(Float32, num_of_kernels, output_height * output_width)

        mul!(output, col_kernel, col_x)

        output = reshape(output, num_of_kernels, output_height, output_width)
        output = permutedims(output, (2, 3, 1))
        return output
    end

backward(::BroadcastedOperator{typeof(convolution)}, x, kernel, g) =
    let
        x = reshape(x, size(x)..., 1, 1)
    
        x_height, x_width, num_of_channels, _ = size(x)
        (filter_height, filter_width, _, num_of_kernels) = size(kernel)

        padding = 0
        stride = 1

        feature_map_height = Int(floor((x_height + 2 * padding - filter_height) / stride)) + 1
        feature_map_width = Int(floor((x_width + 2 * padding - filter_width) / stride)) + 1

        x_with_padding = zeros(x_height + 2 * padding, x_width + 2 * padding, num_of_channels)
        x_with_padding[padding + 1 : end - padding, padding + 1 : end - padding, :] = x

        input_gradient = zeros(x_height + 2 * padding, x_width + 2 * padding, num_of_channels)
        kernel_gradient = zeros(size(kernel))

        for i=1:feature_map_height
            for j=1:feature_map_width
                img_portion = x_with_padding[(i - 1) * stride + 1:(i - 1) * stride + filter_height, (j - 1) * stride + 1 : (j - 1) * stride + filter_width, :, :]

                flatten_img_portion = reshape(img_portion, filter_height * filter_width * num_of_channels, :)
                flatten_kernel = reshape(kernel, filter_height * filter_width * num_of_channels, num_of_kernels)

                local_gradient = reshape(g[i, j, :], num_of_kernels, 1)
                gradient_product = flatten_img_portion * local_gradient'
                gradient_product = reshape(gradient_product, filter_height, filter_width, num_of_channels, num_of_kernels)
                kernel_gradient += gradient_product
                flatten_gradient_product = flatten_kernel * local_gradient
                flatten_gradient_product = reshape(flatten_gradient_product, filter_height, filter_width, num_of_channels, :)
                input_gradient[(i - 1) * stride + 1 : (i - 1) * stride + filter_height, (j - 1) * stride + 1 : (j - 1) * stride + filter_width, :, :] += flatten_gradient_product
            end
        end

        x_gradient = input_gradient[padding + 1 : end - padding, padding + 1 : end-padding, :]

        return x_gradient, kernel_gradient
    end

maxpool2d(x::GraphNode) = BroadcastedOperator(maxpool2d, x)
forward(node::BroadcastedOperator{typeof(maxpool2d)}, x) =
    let
        height, width, num_of_channels = size(x)

        output_height, output_width = trunc(Int, height / 2), trunc(Int, width / 2)
        output = zeros(output_height, output_width, num_of_channels)

        node.cache = CartesianIndex{3}[]

        for i = 1 : num_of_channels
            for j = 1 : output_height
                for k = 1 : output_width
                    maxVal, maxValIndices = findmax(@view x[2*j-1:2*j, 2*k-1:2*k, i])
                    output[j, k, i] = maxVal

                    idx, idy = maxValIndices[1]+2*j- 1 - 1, maxValIndices[2] + 2 * k - 1 - 1
                    
                    push!(node.cache, CartesianIndex(idx, idy, i))
                end
            end
        end
        output
    end
backward(node::BroadcastedOperator{typeof(maxpool2d)}, x, g) =
    let
        output = zeros(size(x))
        output[node.cache] = g
        output
    end

function img2col(img, kernel_height, kernel_width, output_height, output_width)
    img_height, img_width = size(img)
    output = Array{eltype(img)}(undef, kernel_height * kernel_width, output_height * output_width)

    indx = reshape(1:img_height * img_width, img_height, img_width)[1:output_height, 1:output_width]

    for (i, value) in enumerate(indx)
        for j = 0:kernel_width-1
            @views output[(i - 1) * kernel_height * kernel_width + j * kernel_height + 1 : (i - 1) * kernel_height * kernel_width + (j + 1) * kernel_height] = 
            img[value + j * img_height : value + kernel_height - 1 + j * img_height]
        end
    end

    return output
end
    