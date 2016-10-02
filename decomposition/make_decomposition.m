function net_upd = ...
    make_decomposition(net_old, net_orig, lambda_vec, d_, ...
    layer_no, layer_no_orig, imdb, batchNum, batchSize, useGpu)
    % Obtains P and Q - ancillary matrices, in order to obtain weights of the decomposed layers

    layer_size = size(net_old.layers{layer_no}.filters);
    d = layer_size(4); 

    net_orig_cut = net_orig;
    net_orig_cut.layers = net_orig_cut.layers(1:layer_no_orig);
    net_old_cut = net_old;
    net_old_cut.layers = net_old_cut.layers(1:layer_no);

    tic;
    [Y_hat, colsPerBatch] = get_batches(net_old_cut, imdb, batchSize, batchNum);
    [Y_orig, colsPerBatchOrig] = get_batches(net_orig_cut, imdb, batchSize, batchNum);
    fprintf('processing images done\n');
    toc

    y_hat_avg = mean(Y_hat, 2);
    y_orig_avg = mean(Y_orig, 2);

    Abig = zeros(d, d, 'single');
    Bbig = zeros(d, d, 'single');
    
    for i=1:batchNum   
        
        Y_hat_mini = bsxfun(@minus, Y_hat(:, (i-1)*colsPerBatch+1:i*colsPerBatch), y_hat_avg);
        Y_orig_mini = bsxfun(@minus, Y_orig(:, (i-1)*colsPerBatchOrig+1:i*colsPerBatchOrig), y_orig_avg);
  
        Abig = Abig + Y_hat_mini * Y_hat_mini';
        Bbig = Bbig + Y_orig_mini * Y_hat_mini';
        
    end

    [P, Q, M] = get_help_matrices(Abig, Bbig, d_);  %matrix M after the first iteration

    b_new = y_orig_avg - M*y_hat_avg;
 
    t = 0;
    for lambda = lambda_vec
        t = t + 1;
        fprintf('t %d lambda %f\n', t, lambda); 

        M = gpuArray(M);
        b_new = gpuArray(b_new);
        z_avg = zeros(d, 1, 'single');
        ZY_big = zeros(d, d, 'single');
        
        for i=1:batchNum
            
            %Y_hat_mini (without mean), y_hat_avg
            Y_hat_mini_o = Y_hat(:, (i-1)*colsPerBatch+1:i*colsPerBatch);

            %Y_orig_mini (with mean)
            Y_orig_mini = Y_orig(:, (i-1)*colsPerBatchOrig+1:i*colsPerBatchOrig);
            Y_orig_mini = gpuArray(Y_orig_mini);
            Y_hat_mini_o = gpuArray(Y_hat_mini_o);
            
            %Y_, Z0, Z1, Z_new, z_avg
            Y__mini = bsxfun(@plus, M*Y_hat_mini_o, b_new);
            Z0_mini = min(0, Y__mini); 
            clear Y_hat_mini;
            temp_matrix_mini = (lambda * Y__mini + max(0, Y_orig_mini))/(lambda+1); %was Y'!!!
            Z1_mini = max(0, temp_matrix_mini);
            F0_mini = (max(0, Y_orig_mini) - max(0, Z0_mini)).^2 + lambda * (Z0_mini - Y__mini).^2;
            F1_mini = (max(0, Y_orig_mini) - max(0, Z1_mini)).^2 + lambda * (Z1_mini - Y__mini).^2;
            F_mini = F0_mini > F1_mini;
            Z_new_mini = F_mini .* Z1_mini + (~F_mini) .* Z0_mini;
            [~, z_avg] = remove_matrix_mean(Z_new_mini, z_avg, batchNum);
 
        end
        
        for i=1:batchNum
            %Y_hat_mini (without mean), y_hat_avg
            Y_hat_mini = bsxfun(@minus, Y_hat(:, (i-1)*colsPerBatch+1:i*colsPerBatch), y_hat_avg);
            Y_hat_mini_o = Y_hat(:, (i-1)*colsPerBatch+1:i*colsPerBatch);

            %Y_orig_mini (with mean!)
            Y_orig_mini = Y_orig(:, (i-1)*colsPerBatchOrig+1:i*colsPerBatchOrig);
            Y_hat_mini = gpuArray(Y_hat_mini);
            Y_orig_mini = gpuArray(Y_orig_mini);
            Y_hat_mini_o = gpuArray(Y_hat_mini_o);
            
            %Y_, Z0, Z1, Z_new, z_avg
            Y__mini = bsxfun(@plus, M*Y_hat_mini_o, b_new);
            Z0_mini = min(0, Y__mini); 
            temp_matrix_mini = (lambda * Y__mini + max(0, Y_orig_mini))/(lambda+1); %was Y'!!!
            Z1_mini = max(0, temp_matrix_mini);
            F0_mini = (max(0, Y_orig_mini) - max(0, Z0_mini)).^2 + lambda * (Z0_mini - Y__mini).^2;
            F1_mini = (max(0, Y_orig_mini) - max(0, Z1_mini)).^2 + lambda * (Z1_mini - Y__mini).^2;
            F_mini = F0_mini > F1_mini;
            Z_new_mini = F_mini .* Z1_mini + (~F_mini) .* Z0_mini;
            Z_without_mean_mini = bsxfun(@minus,Z_new_mini,z_avg);
            ZY_mini = Z_without_mean_mini * Y_hat_mini';
            ZY_mini = gather(ZY_mini);
            ZY_big = ZY_big + ZY_mini;

        end
        
        [P, Q, M] = get_help_matrices(Abig, ZY_big, d_);
        b_new = z_avg - M*y_hat_avg; %dx1
                
    end

    [W_, W__, b_, b__] = new_net_parameters_creation(net_old, P, Q, d_, layer_no, b_new');
    net_upd = create_new_net(net_old, W_, W__, b_, b__, layer_no);

end
