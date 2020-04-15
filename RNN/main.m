%% Assignment 4

%% Read in the data

book_fname = 'data/goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c'); 
fclose(fid);

book_chars = unique(book_data);
K = length(book_chars);
char_list = cell(K,1);
ind_list = zeros(K,1);
for i = 1:K
    char_list{i} = book_chars(i);
    ind_list(i) = i;
end

char_to_ind = containers.Map(char_list,ind_list);
ind_to_char = containers.Map(ind_list,char_list);
    

%% Set hyper-parameters & initialize the RNN’s parameters
m = 100;
eta = .1;
seq_length = 25;
RNN.b = zeros(m,1);
RNN.c = zeros(K,1);
sig = 0.01;
RNN.U = randn(m, K)*sig;
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;

%% Test gradient

% Implement the forward & backward pass of back-prop
% 
% X_chars = book_data(1:seq_length);
% Y_chars = book_data(2:seq_length+1);
% X = zeros(K, seq_length);
% Y = zeros(K, seq_length);
% for i = 1:seq_length
%     x_ind = char_to_ind(X_chars(i));
%     y_ind = char_to_ind(Y_chars(i));
%     X(x_ind, i) = 1;
%     Y(y_ind, i) = 1;
% end
% 
% h0 = zeros(m, 1);
% [loss, A, H, P] = forwardPass(X, Y, RNN, h0);
% analytical_grads = backwardPass(RNN, A, H, P, X, Y);
% 
% h=1e-4;
% num_grads = ComputeGradsNum(X, Y, RNN, h);
% 
% for f = fieldnames(RNN)'
% 	diffGrads.(f{1}) =  max(abs(analytical_grads.(f{1}) - num_grads.(f{1}))./(abs(num_grads.(f{1}))),[],'all');
% end


%% Training RNN using AdaGrad

% n_epochs = 2; 
% n_updates = n_epochs*size(book_data,2);
n_updates = 100000;
[RNN] = AdaGrad(RNN, char_to_ind, ind_to_char, eta, seq_length, book_data, n_updates);

%% Last text
x= zeros(K,1);
x(char_to_ind('A'))=1;
disp('Here to try with first char A');
disp(synthesize_text( ind_to_char, RNN, zeros(m,1), x, 1000));

%--------------------------------------------------------------------------

%--------------------------------------------------------------------------

