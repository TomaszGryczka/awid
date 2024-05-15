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
        x_height, x_width = size(x)
        filter_height, filter_width, _, num_of_kernels = size(kernel)
        gradient_height, gradient_width, _,  = size(g)

        output_height = x_height - filter_height + 1
        output_width = x_width - filter_width + 1

        dx = zeros(Float32, x_height, x_width)
        dkernel = zeros(Float32, filter_height * filter_width, num_of_kernels)

        img_col = img2col_backpropagation(x, filter_height, filter_width, output_height, output_width)
        g_col = reshape(g, output_height * output_width, num_of_kernels)
        dkernel .+= img_col * g_col
        
        dimg_col = reshape(g, gradient_height*gradient_width, num_of_kernels) * reshape(kernel, filter_height*filter_width, num_of_kernels)'
        col2img(dx, dimg_col, output_height, output_width, filter_height, filter_width)

        dkernel = reshape(dkernel, filter_height, filter_width, 1, num_of_kernels)
        return dx, dkernel
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

function img2col_backpropagation(img, kernel_height, kernel_width, output_height, output_width)
    output = zeros(Float32, kernel_height * kernel_width, output_height * output_width)

    for i = 1:output_height
        for j = 1:output_width
            @views output[:, (i - 1) * output_width + j] .= reshape(img[i:i + kernel_height - 1, j:j + kernel_width - 1], kernel_height * kernel_width)
        end
    end
    return output
end

function col2img(dx, dimg_col, output_height, output_width, filter_height, filter_width)
    for i=1:output_height*output_width-1
        row = dimg_col[i, :]
        h_index = div(i, output_width) + 1
        w_index = mod(i, output_width) + 1
        dx[h_index:h_index+filter_height-1, w_index:w_index+filter_width-1] .+= reshape(row, filter_height, filter_width)
    end
end
    