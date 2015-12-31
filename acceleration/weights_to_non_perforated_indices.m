function [ nonPerforatedIndices ] = weights_to_non_perforated_indices( weights, rate )
% Returns zero-based indices of rate * size(weights, 1) * size(weights, 2) maximum elements from weights matrix.
% Supports batching, although it is not currently used.

sz = [size(weights,1) size(weights,2) 1 size(weights,4)];
toSample = floor(rate * sz(1)*sz(2));

weights = reshape(weights, [sz(1)*sz(2) sz(4)]);
[~, nonPerforatedIndices] = sort(weights, 1, 'descend');
nonPerforatedIndices = nonPerforatedIndices(1:toSample, :);
nonPerforatedIndices = int32(nonPerforatedIndices) - 1;
nonPerforatedIndices = reshape(nonPerforatedIndices, [toSample 1 1 sz(4)]);
nonPerforatedIndices = sort(nonPerforatedIndices);

end
