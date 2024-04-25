include("graph_nodes.jl")

convolution(x::GraphNode, kernel::GraphNode) = BroadcastedOperator(convolution, x, kernel)
forward(::BroadcastedOperator{typeof(convolution)}, x, kernel) =
    let
        x = reshape(x, size(x)..., 1, 1)
        
        x_height, x_width, channels, _ = size(x)
        filter_height, filter_width, _, num_of_kernels = size(kernel)

        padding = 0
        stride = 1

        feature_map_height = Int(floor((x_height + 2 * padding - filter_height) / stride)) + 1
        feature_map_width = Int(floor((x_width + 2 * padding - filter_width) / stride)) + 1

        x_with_padding = zeros(x_height + 2 * padding, x_width + 2 * padding, channels)
        x_with_padding[padding + 1 : end-padding, padding + 1 : end - padding, :] = x

        output = zeros(feature_map_height, feature_map_width, num_of_kernels, 1)

        for i=1:feature_map_height
            for j=1:feature_map_width
                img_portion = x_with_padding[(i - 1) * stride + 1 : (i - 1) * stride + filter_height, (j - 1) * stride + 1 : (j - 1) * stride + filter_width, :, :]

                flatten_img_portion = reshape(img_portion, filter_height * filter_width * channels, :)
                flatten_kernel = reshape(kernel, filter_height * filter_width * channels, num_of_kernels)

                output[i, j, :] = sum(flatten_kernel .* flatten_img_portion, dims = 1)
            end
        end
        return output
    end

backward(::BroadcastedOperator{typeof(convolution)}, x, kernel, g) =
    let
        x = reshape(x, size(x)..., 1, 1)
    
        x_height, x_width, channels, _ = size(x)
        (filter_height, filter_width, _, num_of_kernels) = size(kernel)

        padding = 0
        stride = 1

        feature_map_height = Int(floor((x_height + 2 * padding - filter_height) / stride)) + 1
        feature_map_width = Int(floor((x_width + 2 * padding - filter_width) / stride)) + 1

        x_with_padding = zeros(x_height + 2 * padding, x_width + 2 * padding, channels)
        x_with_padding[padding + 1 : end - padding, padding + 1 : end - padding, :] = x

        input_gradient = zeros(x_height + 2 * padding, x_width + 2 * padding, channels)
        kernel_gradient = zeros(size(kernel))

        for i=1:feature_map_height
            for j=1:feature_map_width
                img_portion = x_with_padding[(i - 1) * stride + 1:(i - 1) * stride + filter_height, (j - 1) * stride + 1 : (j - 1) * stride + filter_width, :, :]

                flatten_img_portion = reshape(img_portion, filter_height * filter_width * channels, :)
                flatten_kernel = reshape(kernel, filter_height * filter_width * channels, num_of_kernels)

                local_gradient = reshape(g[i, j, :], num_of_kernels, 1)
                gradient_product = flatten_img_portion * local_gradient'
                gradient_product = reshape(gradient_product, filter_height, filter_width, channels, num_of_kernels)
                kernel_gradient += gradient_product
                flatten_gradient_product = flatten_kernel * local_gradient
                flatten_gradient_product = reshape(flatten_gradient_product, filter_height, filter_width, channels, :)
                input_gradient[(i - 1) * stride + 1 : (i - 1) * stride + filter_height, (j - 1) * stride + 1 : (j - 1) * stride + filter_width, :, :] += flatten_gradient_product
            end
        end

        x_gradient = input_gradient[padding + 1 : end - padding, padding + 1 : end-padding, :]

        return tuple(x_gradient, kernel_gradient)
    end

maxpool2d(x::GraphNode) = BroadcastedOperator(maxpool2d, x)
forward(node::BroadcastedOperator{typeof(maxpool2d)}, x) =
    let
        height, width, channels = size(x)

        output_height, output_width = trunc(Int, height / 2), trunc(Int, width / 2)
        output = zeros(output_height, output_width, channels)

        indices = CartesianIndex{3}[]

        for i = 1 : channels
            for j = 1 : output_height
                for k = 1 : output_width
                    maxVal, maxValIndices = findmax(@view x[2*j-1:2*j, 2*k-1:2*k, i])
                    output[j, k, i] = maxVal

                    idx, idy = maxValIndices[1]+2*j- 1 - 1, maxValIndices[2] + 2 * k - 1 - 1
                    
                    push!(indices, CartesianIndex(idx, idy, i))
                end
            end
        end
        node.cache = indices
        output
    end
backward(node::BroadcastedOperator{typeof(maxpool2d)}, x, g) =
    let
        output = zeros(size(x))
        output[node.cache] = vcat(g...)
        tuple(output)
    end