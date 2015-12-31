function averageImpacts = weights_average_impact(net, imdb, getBatch, train, convLayers, batchSize)

averageImpacts = cell(length(convLayers), 1);
for l = 1:length(convLayers)
  averageImpacts{l} = 0;
end

res = [];

for t = 1:batchSize:numel(train)
  batch_time = tic ;
    
  % fprintf('weights_average_impact: processing batch %3d of %3d ...', ...
  %         fix(t/batchSize)+1, ceil(numel(train)/batchSize)) ;

  batch = train(t:min(t+batchSize-1, numel(train))) ;
  [im, labels] = getBatch(imdb, batch);
  net.layers{end}.class = labels;
  res = vl_simplenn(net, im, single(1), res, 'disableDropout', true, 'conserveMemory', false, 'sync', true) ;

  for l = 1:length(convLayers)
    impacts = res(convLayers(l) + 1).dzdx .* res(convLayers(l) + 1).x;
    impacts = sum(sum(abs(impacts), 3), 4) / numel(train);

    averageImpacts{l} = averageImpacts{l} + gather(impacts);
  end
  
  batch_time = toc(batch_time) ;
  speed = numel(batch)/batch_time ;
  % fprintf(' %.2f s (%.1f images/s)\n', batch_time, speed) ;
end

% We null out the values for the already perforated positions.
% This is required for iterative pooling impact scheme
for l = 1:length(convLayers)
  if isfield(net.layers{convLayers(l)}, 'nonPerforatedIndices')
    outputSize = net.layers{convLayers(l)}.outputSize;
    nonPerforatedIndices = net.layers{convLayers(l)}.nonPerforatedIndices;
   
    res = zeros(outputSize(1), outputSize(2), 'single');
    res(nonPerforatedIndices(:) + 1) = averageImpacts{l}(:);
    averageImpacts{l} = res;
  end
end

end
