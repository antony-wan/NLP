function P = EvaluateClassifier(X, RNN, h)
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here
n = size(X, 2);
[K,m] = size(RNN.V);
A = zeros(m,n);
H = zeros(m,n);
P = zeros(K,n);
for t = 1:n
    x = X(:,t);
    a = RNN.W*h + RNN.U*x + RNN.b;
    A(:,t) = a;
    
    h = tanh(a);
    H(:,t) = h;
    
    o = RNN.V * h +RNN.c;
    
    p = softmax(o); 
    P(:,t) = p;
end

end

