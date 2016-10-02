function [W_, W__, b_, b__] = new_net_parameters_creation(net_old, P, Q, d_, layer_no, b_upd)
%Computes weights and biases of the new convolutional layers (help matrices are given)
%layer_no is the number of the conv layer which is going to be decomposed

    layer_size = size(net_old.layers{layer_no}.filters);
    k=layer_size(1);c=layer_size(3);
    %W layer
    W = net_old.layers{1, layer_no}.filters;
    W = reshape (W,[],size(W,4))';

    %W' layer
    W_=Q'*W;
    W_=reshape(W_,d_,k,k,c);
    W_=permute(W_,[2,3,4,1]);
    %b_=b(1:d_,:);
    b_=single(zeros(1,d_));

    %P layer
    W__=permute(P,[2,1]);
    W__=reshape(W__,1,1,size(W__,1),size(W__,2));
    b__= b_upd; 
    
end