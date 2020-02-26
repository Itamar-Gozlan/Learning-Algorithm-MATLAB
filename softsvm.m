function w = softsvm(lambda, m, d, Xtrain, Ytrain)
H = [2*lambda*speye(d) zeros(d,m);...
    zeros(m,m+d)];
H = sparse(H);
f = [zeros(1,d) ones(1,m) * 1/m]';
A = -1 * [(diag(Ytrain) * Xtrain) speye(m); zeros(m,d) speye(m,m)];
bb = -1 * [ones(1,m), zeros(1,m)];
w = quadprog(H,f,A,bb);
w = w(1:d)';




