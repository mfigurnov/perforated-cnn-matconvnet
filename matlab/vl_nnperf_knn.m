function y = vl_nnperf_knn(x,maskindices,outindices,weights)

sz = size(x);

idx = maskindices(outindices + 1) + 1;
y = zeros(sz, 'like', x);
x = reshape(x, sz(1)*sz(2), sz(3), sz(4));
for i = 1:size(idx, 3)
  tmp = x(idx(:, :, i), :, :);
  tmp = reshape(tmp, sz);
  if numel(weights) ~= 1
    y = y + bsxfun(@times, tmp, weights(:, :, i));
  else
    y = y + weights .* tmp;
  end
end
