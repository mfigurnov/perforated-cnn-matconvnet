function net_new = ...
    speedup_net_nonlinear(net_orig, net_old, batchNum, batchSize, imdb, getBatch, train, layer_no_vec, ...
    d_vec, lambda_arr, useGpu)

    % Accelerates CNN by the given ranks of conv layers 'd_vec'
    
    if useGpu
        net_old = vl_simplenn_move(net_old, 'gpu');
        net_orig = vl_simplenn_move(net_orig, 'gpu');
    end
    
    rng(0);
    train = train(randperm(numel(train)));
    for i = 1:size(layer_no_vec,2) 
        ['Processing layer ' num2str(layer_no_vec(i))]
        net_new = make_decomposition(net_old, net_orig, lambda_arr , d_vec(i), ...
            layer_no_vec(i)+i-1, layer_no_vec(i), imdb, batchNum(i), batchSize, useGpu);
        net_old = net_new;
        if useGpu
            net_old = vl_simplenn_move(net_old, 'gpu');
        end
    end
    
    %returned net is 'cpu' net
    if useGpu
        net_new = vl_simplenn_move(net_new, 'cpu');
    end
    
end
