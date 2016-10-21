function d__vec = get_rank_vector (speedup_ratio, net_vgg, im3000_vgg, batchNum, batchSize, conv_l_vec, d_max, useGpu)
% Obtains the ranks of the new convolutional layers
% d_max - the max number of neurons among all convs
% res_of_orig_net can be computed if net is known: res = vl_simplenn(net, images)

    C_l_vec = [];
    sigma = zeros(d_max, size(conv_l_vec,2));
    d_vec = [];
    c_vec=[];
    k_vec=[];
    net_vgg = vl_simplenn_move(net_vgg, 'gpu');

    iter=1;
    for layer_no=conv_l_vec
        
        layer_size = size(net_vgg.layers{layer_no}.filters);
        k=layer_size(1);
        c=layer_size(3);
        d=layer_size(4);
        d_vec = [d_vec d];
        c_vec=[c_vec c];
        k_vec=[k_vec k];
        
        YY_big=zeros(d,d);
        for i=1:batchNum
            im100=im3000_vgg(:,:,:,(i-1)*batchSize+1:i*batchSize);
            if useGpu
                im100 = gpuArray(im100);
            end
            res_orig = vl_simplenn(net_vgg, im100, [], [], 'disableDropout', true); 
            %Y matrix and SVD
            Y_hat_mini=res_orig(layer_no+1).x; %AFTER conv now
            Y_hat_mini = permute(Y_hat_mini,[3,1,2,4]);
            Y_hat_mini=reshape(Y_hat_mini,size(Y_hat_mini,1),[]);
            if i ~= batchNum
                clear res_orig;
            end
            y_hat_mini_avg  = mean(Y_hat_mini,2); %kind of batch norm
            %y_hat_avg = y_hat_avg + y_hat_mini_avg./batchNum;
            Y_hat_mini = Y_hat_mini - repmat(y_hat_mini_avg, 1, size(Y_hat_mini,2)); %%% %It's already without mean
            YY_mini=Y_hat_mini*Y_hat_mini';
            YY_big = YY_big+YY_mini;
        end
        w = size(res_orig(layer_no).x,2); %spatial size of the output
        h = size(res_orig(layer_no).x,1); %spatial size of the output
        clear res_orig;

        eigenvalues_vec = eig(YY_big);
        clear YY_big;
        [sortedX,~] = sort(eigenvalues_vec,'descend');
        
        sigma(1:length(sortedX),iter) = gather(sortedX); 
        %matrix of eigenvalues from the biggest ones to the smallest
        
        C_l = d*k^2*c * h*w;
        C_l_vec = [C_l_vec C_l];
        iter= iter+1;
    end
    sigma=sigma';
    C = sum(C_l_vec) / speedup_ratio;
    relative_energy = []; % array with dim (1, #of convolutions)
    iter=0;
    complexity_reduction_vec = C_l_vec .*( (d_vec+(k.^2).*c)./(d_vec.*(k.^2).*c));
    d__vec=d_vec; % initialization
    while (sum(((d__vec.*(k.^2) .*c)+d_vec.*d__vec)./(d_vec.*(k.^2) .*c) .*C_l_vec) > C)

        linearindex = sub2ind(size(sigma), 1:size(conv_l_vec,2), d__vec); %indices of sigma(l,d_)
        % orientation in matrix of sigmas with zeros (padded sigma matrix)
        sum_eig_vec = [];
        for layer_no=1:size(conv_l_vec,2)
            sum_eig = sum(sigma(layer_no,1:d__vec(layer_no)));
            sum_eig_vec = [sum_eig_vec sum_eig];
        end
        measure_vec = (sigma(linearindex)./sum_eig_vec)./complexity_reduction_vec;
        [~,min_idx] = min(measure_vec);
        iter = iter + 1;
        d__vec(min_idx) = d__vec(min_idx)-1;
        %upd d_vec
    end
end
