function [net_upd] = ...
    speedup_layer_linear(net_old, net_orig,...
    batchNum, batchSize, im3000_vgg, layer_no, layer_no_orig,...
    d_vec, byRatio, useGpu)
    % Speedups the layer

    % Layer's parameters
    layer_size = size(net_orig.layers{1,layer_no_orig}.filters);
    d = layer_size(4); k1=layer_size(1); k2=layer_size(2); c=layer_size(3);
    if byRatio
        d_ = floor((d*k1*k2* c) / (d_vec *(k1*k2*c+d)));
    else
        d_ = d_vec;
    end

    % Initialization
    Abig=zeros(d,d);
    Bbig = zeros(d,d);
    y_hat_avg = zeros(d, 1);
    y_orig_avg = zeros(d, 1);
    
    % [A B C]* [A B C]',  A*A + B*B + C*C (batch implementation)
    for i=1:batchNum
        im100=im3000_vgg(:,:,:,(i-1)*batchSize+1:i*batchSize);
        if useGpu
            im100 = gpuArray(im100);
        end
        res = vl_simplenn(net_old, im100, [], [], 'disableDropout', true); 
        
        %Y matrix and SVD
        Y_hat_mini=res(layer_no+1).x; %AFTER conv
        clear res;
        
        if isfield(net_old.layers{1,layer_no}, 'interpolationIndicesOut')
            outInd = net_old.layers{1,layer_no}.interpolationIndicesOut;
            Y_hat_mini_2 = gpuArray(zeros(size( outInd,1), size( outInd,2) , d, size(Y_hat_mini,4)));
            for neuronnum = 1:d
                for imnum = 1:size(Y_hat_mini,4)
                    Y_temp = Y_hat_mini(:,:,neuronnum, imnum);
                    
                    Y_hat_mini_2 (:,:,neuronnum, imnum) = ...
                        Y_temp( outInd + 1);
                    clear Y_temp;
                end
            end
            clear outInd;
            clear Y_hat_mini;
            Y_hat_mini = Y_hat_mini_2;
            clear Y_hat_mini_2;
        end
        
        Y_hat_mini = permute(Y_hat_mini,[3,1,2,4]);
        Y_hat_mini=reshape(Y_hat_mini,size(Y_hat_mini,1),[]);
        y_hat_mini_avg  = mean(Y_hat_mini,2);
        y_hat_avg = y_hat_avg + y_hat_mini_avg./batchNum;
        Y_hat_mini = Y_hat_mini - repmat(y_hat_mini_avg, 1, size(Y_hat_mini,2)); 
        clear y_hat_mini_avg;
       
        res_orig = vl_simplenn(net_orig, im100, [], [], 'disableDropout', true); 
        clear im100;
        
        %Y matrix and SVD
        Y_orig_mini=res_orig(layer_no_orig+1).x; %After conv layer
        clear res_orig;
        Y_orig_mini = permute(Y_orig_mini,[3,1,2,4]);
        Y_orig_mini=reshape(Y_orig_mini,size(Y_orig_mini,1),[]);
        y_orig_mini_avg  = mean(Y_orig_mini,2); %kind of batch norm
        y_orig_avg = y_orig_avg + y_orig_mini_avg./batchNum;
        Y_orig_mini = Y_orig_mini - repmat(y_orig_mini_avg, 1, size(Y_orig_mini,2)); %without batch mean
        clear y_orig_mini_avg;
        A=Y_hat_mini*Y_hat_mini';
        Abig = Abig+A;
        clear A;
        
        B = Y_orig_mini * Y_hat_mini';
        clear Y_hat_mini;
        clear Y_orig_mini;
        Bbig = Bbig + B;
        clear B;
    end
 
    M_hat = Bbig*(Abig)^(-1);
    A_tilde = M_hat * (Abig)^(1/2);
    [P,S,Q] = svd(A_tilde); 
    %S - from the biggest eigenvalues to the smallest ones
    %sum(sum(P'*P == 1)), sum(sum(Q'*Q == 1))
    U = P;
    V = (Abig)^(-1/2) * Q;
    S_d_ = S(1:d_,1:d_);
    U_d_ = U(:,1:d_);
    V_d_ = V(:,1:d_);
    M = U_d_ * S_d_ * V_d_'; % matrix M is created 
    b_new = (y_orig_avg - M*y_hat_avg)'; 
    P = U_d_ * S_d_^(1/2);
    Q = V_d_ * S_d_^(1/2);
    
    % Create a new CNN
    [W_, W__, b_, b__] = ...
       new_net_parameters_creation(net_old, P, Q, d_, layer_no, b_new);
    net_upd = create_new_net(net_old, W_, W__, b_, b__, layer_no);
    net_upd = net_rm_indices(net_upd, layer_no);

end