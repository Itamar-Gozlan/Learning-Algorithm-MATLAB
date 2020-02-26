function classifier = learnknn(k, d, m, Xtrain, Ytrain)
[C,~,Ytrain] = unique(Ytrain,'stable');

classifier.k = k;
classifier.d = d;
classifier.m = m;
classifier.Xtrain = Xtrain;
classifier.Ytrain = C;
classifier.labels = Ytrain;

