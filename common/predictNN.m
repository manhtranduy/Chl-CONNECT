function output = predictNN(weights_and_biases, X)
    % Perform matrix multiplication, add bias, and apply ReLU activation
    for i = 1:length(weights_and_biases)
        weights = weights_and_biases{i, 1};
        biases = weights_and_biases{i, 2};
        
        % Matrix multiplication and add bias
        X = X* weights' + biases';
        
        % Apply ReLU activation function for all but the last layer
        if i < length(weights_and_biases)  % ReLU for hidden layers
            X = relu(X);
        end
    end
    output = X(:);  % Flatten the output
end

