function net_new = ...
    speedup_net_linear(net_orig, net_old, batchNum, batchSize, im3000_vgg, layer_no_vec, ...
    d_vec, byRatio, useGpu)
    %speedups CNN

    if useGpu
        net_old = vl_simplenn_move(net_old, 'gpu');
        net_orig = vl_simplenn_move(net_orig, 'gpu');
    end
    
    for i = 1:size(layer_no_vec,2) 
        i   
        if byRatio
            net_new = speedup_layer_linear(net_old, net_orig,...
                batchNum, batchSize, im3000_vgg, layer_no_vec(i)+i-1, layer_no_vec(i),...
                d_vec, byRatio, useGpu);
        else
            net_new = speedup_layer_linear(net_old, net_orig,...
                batchNum, batchSize, im3000_vgg, layer_no_vec(i)+i-1, layer_no_vec(i),...
                d_vec(i), byRatio, useGpu);
        end
        net_old = net_new;
        if useGpu
            net_old = vl_simplenn_move(net_old, 'gpu');
        end
    end
    
    %returned net is 'cpu'
    if useGpu
        net_new = vl_simplenn_move(net_new, 'cpu');
    end
    
end