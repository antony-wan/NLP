function [grads] = backwardPass(RNN, A, H, P, X, Y)

[K,m] = size(RNN.V);
n = size(X,2);

grads.V = zeros(K,m);
grads.W = zeros(m,m);
grads.U = zeros(m,K);
grads.b = zeros(m,1);
grads.c = zeros(K,1);

G = -(Y - P)';

% grad V
for t = 1:n
    grads.V = grads.V + G(t,:)'*H(:,t)';
end

% grad c
for t = 1:n
    grads.c = grads.c + G(t,:)';
end

% grad h
dL_h = zeros(n,m);
dL_a = zeros(n,m);

dL_h(n,:) = G(n,:)*RNN.V;
dL_a(n,:) = dL_h(n,:)*diag(1-tanh(A(:,n)).^2);

for t = n-1:-1:1
    dL_h(t,:) = G(t,:)*RNN.V + dL_a(t+1,:)*RNN.W ;
    dL_a(t,:) = dL_h(t,:)*diag(1-tanh(A(:,t)).^2);   
end

% grad W
%%Considering that h0 = zeros(m,1);
for t = 2:n
    grads.W = grads.W + dL_a(t,:)'*H(:,t-1)';
end

% grad U
for t = 1:n
    grads.U = grads.U + dL_a(t,:)'*X(:,t)';
end

% grad b
for t = 1:n
    grads.b = grads.b + dL_a(t,:)';
end

end
