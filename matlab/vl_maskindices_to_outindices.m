function [ outindices ] = vl_maskindices_to_outindices( maskindices, sz )

mask = false(sz);
mask(maskindices + 1) = true;
[row, col] = find(mask);
nonperf = [row col];
[row, col] = ind2sub([sz(1) sz(2)], 1:(sz(1)*sz(2)));
grid = [row' col'];

mdl = KDTreeSearcher(nonperf);
idx = mdl.knnsearch(grid, 'IncludeTies', true);

% rng(0);
minidx = zeros(size(idx, 1), 1);
for i = 1:size(idx, 1)
  minidx(i) = idx{i}(randperm(numel(idx{i}), 1));
end
minidx = reshape(minidx, sz);
outindices = int32(minidx) - 1;

end
