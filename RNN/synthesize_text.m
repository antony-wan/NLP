function [text] = synthesize_text( ind_to_char, RNN, h, x, n)

[K,m] = size(RNN.V);
sampled_chars = zeros(n,1);
text = '';

for i = 1:n

    a = RNN.W*h + RNN.U*x + RNN.b;
    h = tanh(a);
    o = RNN.V*h + RNN.c;
    p = softmax(o);

    % taking the next sample
    cp = cumsum(p);
    a = rand;
    ixs = find(cp-a >0);
    ii = ixs(1);

    % xnext
    x = zeros(K,1);
    x(ii,1) = 1;
    sampled_chars(i) = ii;
    text(i) = ind_to_char(ii);
end



