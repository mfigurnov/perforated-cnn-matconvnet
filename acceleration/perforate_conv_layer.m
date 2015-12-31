function net = perforate_conv_layer(net, rate, perforationType, convLayerData)

i = convLayerData.index;

assert(strcmp(net.layers{i}.type, 'conv'));
assert(strcmp(net.layers{convLayerData.nextPoolingIndex}.type, 'pool'));

outputSize = convLayerData.outputSize;
sz = outputSize(1:2);

% rng(i);

if perforationType == PerforationType.Uniform
  weights = rand(sz);
elseif perforationType == PerforationType.Grid || perforationType == PerforationType.FractionalStride
  % assumes square activations for simplicity
  numPointsSq = rate * prod(sz);
  numPointsX = floor(sqrt(numPointsSq));
  numPointsY = numPointsX;
  rate = (numPointsX * numPointsY) / prod(sz);
  weights = weights_grid(sz, [numPointsX numPointsY]);
elseif perforationType == PerforationType.Structure
  weights = convLayerData.structureWeights;
  % randomly break ties
  weights = weights + rand(size(weights)) * 1e-5;
elseif perforationType == PerforationType.Impact
  weights = convLayerData.impacts;
elseif perforationType == PerforationType.IterativeImpact
  rateIdx = find(convLayerData.iterativeImpactRates == rate);
  assert(~isempty(rateIdx));
  weights = convLayerData.iterativeImpacts{rateIdx};
  assert(~isempty(weights));
else
  error('Unknown perforationType');
end

if perforationType == PerforationType.Grid
  net.layers{i}.nonPerforatedIndices = int32(find(weights(:))) - 1;
else
  net.layers{i}.nonPerforatedIndices = weights_to_non_perforated_indices(weights, rate);
end

if perforationType ~= PerforationType.FractionalStride
  % next layer should use appropriate indices for unmasking
  interpolationIndices = non_perforated_indices_to_intepolation_indices(net.layers{i}.nonPerforatedIndices, sz);
  net.layers{i}.outputSize = outputSize;
  net.layers{convLayerData.nextLayer}.interpolationIndicesIn = interpolationIndices;
  net.layers{i}.interpolationIndicesOut = interpolationIndices;
else
  net.layers{i}.outputShape = [numPointsX numPointsY];
end

net.layers{i}.rate = rate;

end
