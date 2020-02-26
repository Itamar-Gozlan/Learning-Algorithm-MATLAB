function w = ridge_reg(lambda, X, Y, train_size)
    X_M = X(:,1:train_size);
    Y_M = Y(1:train_size);
    res = X_M*X_M';
    if lambda == 0
        epsilon = 0.01*eye(size(res));
    else
        epsilon = 0;
    end
    w = inv(res + epsilon + lambda*eye(size(res)))*X_M*Y_M;
end