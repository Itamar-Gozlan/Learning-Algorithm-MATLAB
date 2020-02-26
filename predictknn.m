function Ytestprediction = predictknn(classifier, n, Xtest)

Ytestprediction = zeros(n,1);
Y = classifier.labels;
X = classifier.Xtrain;
m = classifier.m;
k = classifier.k;
C = classifier.Ytrain;
for i=1:n
    Xnew = Xtest(i,:);
    A = repmat(Xnew, m, 1) - X;
    distances = sqrt(sum(A.^2,2));
    [~,I] = sort(distances);
    %Xnearest = X(I(1:k), :);
    Ynearest = C(Y(I(1:k)));
    count = zeros(size(C,1),1);
    %casting a vote
    for j=1:k
        idx = find(C==Ynearest(j));
        count(idx) = count(idx) + 1;
    end
    %finding max idx
    max_idx = 0;
    max_num = -1;
    for j = 1:size(C,1)
        if(count(j) > max_num)
            max_idx = j;
            max_num = count(j);
        end
    end
    Ytestprediction(i) = C(max_idx);
end