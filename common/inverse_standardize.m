function Z = inverse_standardize(X, mean, std)
    Z = X.* std' + mean';