function [ interpolationIndices ] = non_perforated_indices_to_intepolation_indices( ...
  nonPerforatedIndices, outputSize )
% Finds nearest neighbor for each output spatial position.
% For non-perforated positions that is the position itself.
% The ties are broken randomly.

mask = false(outputSize);
mask(nonPerforatedIndices + 1) = true;
[row, col] = find(mask);
nonPerforatedPositions = [row col];
[row, col] = ind2sub([outputSize(1) outputSize(2)], 1:(outputSize(1)*outputSize(2)));
allPositions = [row' col'];

searcher = KDTreeSearcher(nonPerforatedPositions);
nnIndices = searcher.knnsearch(allPositions, 'IncludeTies', true);

% rng(0);
% Selects a random nearest neighbor. This leaves non-perforated positions in place,
% since they have unique nearest neighbor.
nearestIndex = zeros(size(nnIndices, 1), 1);
for i = 1:size(nnIndices, 1)
  nearestIndex(i) = nnIndices{i}(randperm(numel(nnIndices{i}), 1));
end
nearestIndex = reshape(nearestIndex, outputSize);
interpolationIndices = int32(nearestIndex) - 1;

end
