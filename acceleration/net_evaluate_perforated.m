function net_evaluate_perforated(getData, netPath, batchSize, numRepeat, useGpu, useGpuTimings, prefetch)

if useGpuTimings
  deviceName = 'gpu';
else
  deviceName = 'cpu';
end

fprintf('Device name: %s\n', deviceName);
fprintf('Network: %s\n', netPath);

% [net, imdb, getBatch, ~, val] = getData(useGpu);
[netTime, imdbTime, getBatchTime] = getData(useGpuTimings);

inputSizesData = net_input_sizes(netTime, imdbTime, getBatchTime, batchSize, useGpuTimings);
convLayersData = conv_layers(netTime, inputSizesData);

[ ~, originalTime, ~, mflops, mem ] = net_total_time(convLayersData, netTime, imdbTime, getBatchTime, batchSize, numRepeat, true);
fprintf('Original time: %.4f\n', originalTime);

clear net netTime

load(netPath, 'net');
if useGpu
  net = vl_simplenn_move(net, 'gpu');
else
  net = vl_simplenn_move(net, 'cpu');
end
netTime = vl_simplenn_move(net, deviceName);

[ ~, perforatedTime, ~, mflopsPerf, memPerf ] = net_total_time(convLayersData, netTime, imdbTime, getBatchTime, batchSize, numRepeat, true);

fprintf('time %.4f (%.2fx)\n', perforatedTime, originalTime / perforatedTime);
fprintf('MFLOPS reduction: %.4f\n', mflops / mflopsPerf);
fprintf('Memory reduction: %.4f\n', mem / memPerf);

perforatedInfo = cnn_validate(net, imdb, getBatch, val, batchSize, useGpu, prefetch);

fprintf('err %.4f err5 %.4f\n', perforatedInfo.error * 100, perforatedInfo.topFiveError * 100);

end
