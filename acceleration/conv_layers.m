function [ convLayersData ] = conv_layers( net, inputSizesData )
% Finds all non-1x1 convolutional layers and logs some information about them:
% supported perforation types, next layer index, next pooling layer index, ...

isnot1x1conv = @(l) (isequal(l.type, 'conv') && ...
  ~(size(l.filters, 1) == 1 && size(l.filters, 2) == 1 && ...
  isequal(l.stride, [1 1]) && isequal(l.pad, [0 0 0 0])));
ispooling = @(l) isequal(l.type, 'pool');

convLayersData = cell(0, 1);
for i = 1:length(net.layers)
  if ~isnot1x1conv(net.layers{i})
    continue;
  end
  
  foundPooling = false;
  for nextPoolingIndex = i+1:length(net.layers)
    if ispooling(net.layers{nextPoolingIndex})
      foundPooling = true;
      break;
    end
  end
  
  % not followed by pooling, assume it's a fully-connected layer
  if ~foundPooling
    continue;
  end

  for nextLayer = i+1:length(net.layers)
    if isnot1x1conv(net.layers{nextLayer}) || ispooling(net.layers{nextLayer})
      break;
    end
  end
  
  if nextPoolingIndex == nextLayer
    perforationTypes = [PerforationType.Uniform PerforationType.Grid PerforationType.Structure ...
      PerforationType.Impact PerforationType.IterativeImpact];
  else
    perforationTypes = [PerforationType.Uniform PerforationType.Grid ...
      PerforationType.Impact PerforationType.IterativeImpact];
  end
  
  convLayersData{end+1} = struct('index', i, ...
    'nextPoolingIndex', nextPoolingIndex, ...
    'nextLayer', nextLayer, ...
    'perforationTypes', perforationTypes, ...
    'inputSize', inputSizesData(i, :), ...
    'outputSize', inputSizesData(i+1, :));
end

end
