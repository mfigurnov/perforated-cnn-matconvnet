function [ inputSizesData ] = net_input_sizes( net, imdb, getBatch, batchSize, useGpu )
% Records shapes (sizes) of all intermediate activations of a CNN

net.layers{end} = struct('type', 'softmax');
if useGpu
  net = vl_simplenn_move(net, 'gpu') ;
else
  net = vl_simplenn_move(net, 'cpu');
end

im = getBatch(imdb, 1:batchSize);
if useGpu
  im = gpuArray(im);
end

res = vl_simplenn(net, im, [], [], 'disableDropout', true, 'conserveMemory', false, 'sync', true);

inputSizesData = zeros(length(net.layers), 4);

for i = 1:length(net.layers)
  inputSizesData(i, :) = size(res(i).x);
end

end
