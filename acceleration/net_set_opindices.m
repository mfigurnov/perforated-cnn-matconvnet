function [ net ] = net_set_opindices(net, inputSizesData, useGpu)

for i = 1:length(net.layers)
  l = net.layers{i};

  % interpolation indices for the inputs (outputs of the previous layer)!
  interpolationIndicesIn = vl_getfielddefault(l, 'interpolationIndicesIn');

  nonPerforatedIndices = vl_getfielddefault(l, 'nonPerforatedIndices');
  
  switch l.type
    case 'pool'
      l.opindices = vl_nnpoolidx(inputSizesData(i,:), l.pool, 'method', l.method, 'pad', l.pad, ...
        'stride', l.stride, 'inindices', interpolationIndicesIn);
      
      % CPU and GPU implementations use different order of opindices tensor to improve memory coalescing
      if useGpu
        l.opindices = gpuArray(permute(l.opindices, [2 3 1]));
      end
    case 'conv'
      % Skip fully-connected layers
      if inputSizesData(i+1, 1) == 1 && inputSizesData(i+1, 2) == 1
        continue;
      end
      % Skip 1x1 convolutions
      if size(l.filters, 1) == 1 && size(l.filters, 2) == 1 && isequal(l.stride, [1 1]) && isequal(l.pad, [0 0 0 0])
        continue;
      end
      
      l.opindices = vl_nnconvidx(inputSizesData(i,:), size(l.filters), 'pad', l.pad, 'stride', l.stride, ...
        'inindices', interpolationIndicesIn, 'maskindices', nonPerforatedIndices);

      if useGpu
        l.opindices = gpuArray(l.opindices);
      end
  end
  
  net.layers{i} = l;
end

end
