function [RNN] = AdaGrad(RNN, char_to_ind, ind_to_char, eta, seq_length, book_data, nb_updates)

[K,m] = size(RNN.V);
smooth_loss = 100;
m_theta.b = zeros(m, 1);
m_theta.c = zeros(K, 1);
m_theta.U = zeros(m, K);
m_theta.W = zeros(m, m);
m_theta.V = zeros(K, m);
e = 1;
n_plot = round(nb_updates/100);
loss_plot = zeros(n_plot,1);
plot_ind = 0;
hprev = zeros(m,1);

for update_step = 1:nb_updates
    
    if e>length(book_data)-seq_length-1
        e = 1;
        hprev = zeros(m,1);
    end
    
    X_chars = book_data(e:e+seq_length-1);
    Y_chars = book_data(e+1:e+seq_length);
    X = zeros(K, seq_length);
    Y = zeros(K, seq_length);
    for i = 1:seq_length
        x_ind = char_to_ind(X_chars(i));
        y_ind = char_to_ind(Y_chars(i));
        X(x_ind, i) = 1;
        Y(y_ind, i) = 1;
    end
    [loss, A, H, P] = forwardPass(X, Y, RNN, hprev);
    
    [grads] = backwardPass(RNN, A, H, P, X, Y);
    
%Clip gradients to avoid the exploding gradient problem.
    for f = fieldnames(grads)'
        grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
    end
    
    for f = fieldnames(RNN)'
        m_theta.(f{1}) =  m_theta.(f{1}) + grads.(f{1}).^2;
        RNN.(f{1}) = RNN.(f{1}) - eta*grads.(f{1})./(sqrt( m_theta.(f{1}) + eps ));
    end
    
    smooth_loss = .999* smooth_loss + .001 * loss;
    
    hprev = H(:,seq_length);
    e = e + seq_length;
    
    if mod(update_step,10000) == 0

        result = ['iter = ', num2str(update_step), '    |   smooth_loss = ', num2str(smooth_loss)];
        disp(result)
        figure(1);
        plot_ind = plot_ind+1;
        loss_plot(plot_ind) = smooth_loss;
        plot((1:plot_ind)*100,loss_plot(1:plot_ind));
        xlabel('updates');
        ylabel('smooth\_loss');
    end
    
    if mod(update_step, 10000) == 0
        disp('text = ');
        disp(synthesize_text(ind_to_char, RNN, hprev, X(:,1), 200));
        disp('===============================================')

    end
        
end
disp('Last text = ');
disp(synthesize_text(ind_to_char, RNN, hprev, X(:,1),200));
end