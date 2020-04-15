function loss = ComputeLoss(X,Y,RNN,h)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
[P] = EvaluateClassifier(X,RNN,h);
n = size(X,2);
loss = 0;
for t = 1:n
    y = Y(:,t);
    p = P(:,t);
    loss = loss -log(y'*p);
end

