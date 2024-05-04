function x = relu(x)
    % ReLU activation function
    x(x < 0) = 0;
end