function [Y_hat, colsPerBatch] = get_batches(net_cut, imdb, batchSize, batchNum)

    for i=1:batchNum
        %batch = train((i-1)*batchSize+1:i*batchSize) ;
        %im = im;
        im100=imdb.data(:,:,:,(i-1)*batchSize+1:i*batchSize);
        im = gpuArray(im100);
        res = vl_simplenn(net_cut, im, [], [], 'disableDropout', true, 'conserveMemory', true);
        Y_hat_mini = gather(res(end).x); %AFTER convolutional layer
        size(Y_hat_mini)
        Y_hat_mini = permute(Y_hat_mini,[3,1,2,4]);
        Y_hat_mini = reshape(Y_hat_mini, size(Y_hat_mini,1), []);

        if i == 1
            colsPerBatch = size(Y_hat_mini, 2);
            Y_hat = zeros(size(Y_hat_mini,1), size(Y_hat_mini,2) * batchNum, 'single');
        end
        
        Y_hat(:, (i-1)*size(Y_hat_mini,2)+1:i*size(Y_hat_mini,2)) = Y_hat_mini;
    end
end