function [loss, A, H, P] = forwardPass(X, Y, RNN, h)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
n = size(X, 2);
[K,m] = size(RNN.V);
A = zeros(m,n);
H = zeros(m,n);
P = zeros(K,n);
h0 = h;

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

loss = ComputeLoss(X,Y,RNN,h0);

end

