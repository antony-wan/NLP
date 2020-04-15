function [p] = softmax(s)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
p = exp(s)/sum(exp(s));
end

