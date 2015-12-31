function [ convLayersTime, totalNetTime, allLayersTime, totalMflops, resMem ] = ...
  net_total_time(convLayersData, net, imdb, getBatch, batchSize, numRepeat, additionalInfo)

if nargin < 7
  additionalInfo = false;
end

net.layers{end}.type = 'softmax';
im = getBatch(imdb, 1:batchSize);

res = vl_simplenn(net, im, [], [], 'disableDropout', true, 'conserveMemory', false, 'sync', true);

t = zeros(length(net.layers)+1, numRepeat);
for i = 1:numRepeat
  res = vl_simplenn(net, im, [], res, 'disableDropout', true, 'conserveMemory', false, 'sync', true);
  t(:, i) = cat(1, res.time);
end
meanTime = mean(t, 2);

if additionalInfo
  [~, totalMflops, resMem] = vl_simplenn_display(net, res);
else
  totalMflops = [];
  resMem = [];
end

convIndices = zeros(length(convLayersData), 1);
for i = 1:length(convLayersData)
  convIndices(i) = convLayersData{i}.index;
end

allLayersTime = meanTime;
convLayersTime = meanTime(convIndices);
totalNetTime = sum(meanTime);

end
