function [ weights ] = weights_pooling_structure( height, width, pool, varargin )

opts.stride = 1 ;
opts.pad = 0 ;

if isstr(varargin{1}) && strcmpi(varargin{1}, 'verbose')
  opts = vl_argparse(opts, varargin(2:end));
else
  opts = vl_argparse(opts, varargin);
end

if length(pool) == 1
    windowHeight = pool;
    windowWidth = pool;
elseif length(pool) == 2
    windowHeight = pool(1);
    windowWidth = pool(2);
else
    error('SIZE has neither one nor two elements.');
end

if length(opts.stride) == 1
    strideY = opts.stride;
    strideX = opts.stride;
elseif length(opts.stride) == 2
    strideY = opts.stride(1);
    strideX = opts.stride(2);
else
    error('STRIDE has neither one nor two elements.');
end

if strideX < 1 || strideY < 1
    error('At least one element of STRIDE is smaller than one.');
end

if length(opts.pad) == 1
    padTop = opts.pad;
    padBottom = opts.pad;
    padLeft = opts.pad;
    padRight = opts.pad;
elseif length(opts.pad) == 4
    padTop = opts.pad(1);
    padBottom = opts.pad(2);
    padLeft = opts.pad(3);
    padRight = opts.pad(4);
else
    error('PAD has neither one nor four elements.');
end

if height < windowHeight || width < windowWidth
    error('Pooling SIZE is larger than the DATA.');
end

if windowHeight == 0 || windowWidth == 0
    error('A dimension of the pooling SIZE is void.');
end

if strideX == 0 || strideY == 0
    error('An element of STRIDE is zero.');
end

if padLeft < 0 || padRight < 0 || padTop < 0 || padBottom < 0
    error('An element of PAD is negative.');
end

if padLeft >= windowWidth || padRight >= windowWidth || padTop >= windowHeight  || padBottom >= windowHeight
    error('A padding value is larger or equal than the size of the pooling window.');
end

pooledWidth = floor((width + (padLeft + padRight) - windowWidth)/strideX) + 1 ;
pooledHeight = floor((height + (padTop + padBottom) - windowHeight)/strideY) + 1 ;

weights = zeros(height, width, 'single');

for y = 1:pooledHeight
    for x = 1:pooledWidth
        x1 = (x-1) * strideX - padLeft + 1;
        y1 = (y-1) * strideY - padTop + 1;
        x2 = min(x1 + windowWidth - 1, width);
        y2 = min(y1 + windowHeight - 1, height);
        x1 = max(x1, 1);
        y1 = max(y1, 1);

        for py = y1:y2
            for px = x1:x2
                weights(py, px) = weights(py, px) + 1;
            end
        end
    end
end

end
