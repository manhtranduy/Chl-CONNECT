function Z = standardize(X, means, stds)
    % Subtract the mean and divide by the standard deviation
    Z = (X - means') ./ stds';
end

