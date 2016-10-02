function [Y_hat_mini, y_hat_avg] = remove_matrix_mean(Y_hat_mini, y_hat_avg, batchNum)
%Removes mean from the matrix (average is given)
    y_hat_mini_avg = mean(Y_hat_mini,2); %kind of batch norm
    y_hat_avg = y_hat_avg + y_hat_mini_avg./batchNum;
    Y_hat_mini = bsxfun(@minus, Y_hat_mini, y_hat_mini_avg);
end