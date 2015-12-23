function y = vl_nnperf_zeros(x,maskindices,varargin)

backMode = numel(varargin) > 0 ;
if backMode
  dzdy = varargin{1} ;
end

sz = size(x);

mask = false(sz(1), sz(2));
mask(maskindices + 1) = true;

% do job
if ~backMode
  y = bsxfun(@times, x, mask) ;
else
  y = bsxfun(@times, dzdy, mask) ;
end
