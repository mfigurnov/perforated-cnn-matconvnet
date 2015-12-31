function greedyProfile = net_greedy_perforation(...
  getData, rates, dataDir, expDir, imdbPath, modelPath, batchSizeTrain, batchSizeVal, useGpu, useGpuTimings, prefetch, lossFieldName, ...
  numSteps, trainImpactSize, trainSize, valSize, validateOnTrain, numRepeat, numRepeatMask, ...
  perforationTypes)
% Greedily perforates all layers of a CNN

if useGpuTimings
  deviceName = 'gpu';
else
  deviceName = 'cpu';
end

fprintf('Device name: %s\n', deviceName);

if ~exist(expDir)
  mkdir(expDir);
end

folder = fullfile(expDir, ['greedy_perforation_' deviceName]);
if ~exist(folder)
  mkdir(folder);
end

[net, imdb, getBatch, train, val, getBatchTrain] = getData(useGpu,'dataDir',dataDir,'expDir',expDir,'imdbPath',imdbPath,'modelPath',modelPath);
[netTime, imdbTime, getBatchTime] = getData(useGpuTimings,'dataDir',dataDir,'expDir',expDir,'imdbPath',imdbPath,'modelPath',modelPath);

trainOriginal = train;
valOriginal = val;

if trainImpactSize
  rng(0);
  trainImpact = trainOriginal(randperm(numel(trainOriginal), trainImpactSize));
else
  trainImpact = trainOriginal;
end

if trainSize
  rng(1);
  train = trainOriginal(randperm(numel(trainOriginal), trainSize));
end

if valSize
  rng(2);
  if validateOnTrain
    val = trainOriginal(randperm(numel(trainOriginal), valSize));
  else
    val = valOriginal(randperm(numel(valOriginal), valSize));
  end
end

inputSizesData = net_input_sizes(net, imdb, getBatch, 2, useGpu);
convLayersData = conv_layers(net, inputSizesData);

infoValOriginal = cnn_validate(net, imdb, getBatch, val, batchSizeVal, useGpu, prefetch);
originalLoss = infoValOriginal.(lossFieldName);
[ ~, netTotalTimeOriginal, ~ ] = net_total_time(convLayersData, netTime, imdbTime, getBatchTime, batchSizeVal, numRepeat*2);
fprintf('original time %.4f s original loss %.4f\n', netTotalTimeOriginal, originalLoss);

numConvLayers = length(convLayersData);

convLayers = zeros(numConvLayers, 1);
for convIdx = 1:numConvLayers
  convLayers(convIdx) = convLayersData{convIdx}.index;
end

% shouldContinue = exist([folder '.mat'], 'file');
% if shouldContinue
%   fprintf('Continuing...\n');
%   load([folder '.mat']);
%   for firstStepIdx = 1:length(greedyProfile)
%     if isempty(greedyProfile{firstStepIdx})
%       break;
%     end
%   end
%   load(fullfile(folder, ['net_' num2str(firstStepIdx - 1) '.mat']));
%   previousBestInfo = greedyProfile{firstStepIdx - 1};
% else
  firstStepIdx = 1;

  for convIdx = 1:numConvLayers
    convLayersData{convIdx}.perforationTypes = perforationTypes;

    if any(perforationTypes == PerforationType.Structure)
      l = net.layers{convLayersData{convIdx}.nextPoolingIndex};
      convLayersData{convIdx}.structureWeights = weights_pooling_structure(...
        convLayersData{convIdx}.outputSize(1), convLayersData{convIdx}.outputSize(2), l.pool, 'pad', l.pad, 'stride', l.stride);
      clear l
    end

    convLayersData{convIdx}.iterativeImpactRates = rates;
    convLayersData{convIdx}.iterativeImpacts = cell(length(rates) + 1, 1);
  end

  % calculate iterative impacts
  if any(perforationTypes == PerforationType.IterativeImpact)
    fprintf('Recalculating impacts... ');
    averageImpacts = weights_average_impact(net, imdb, getBatch, trainImpact, convLayers, batchSizeTrain);
    for convIdx = 1:numConvLayers
      convLayersData{convIdx}.impacts = averageImpacts{convIdx};
      convLayersData{convIdx}.iterativeImpacts{2} = averageImpacts{convIdx};
    end
    fprintf('done\n');
  end

  greedyProfile = cell(numSteps, 1);
  previousBestInfo = struct('perfIdx', ones(1, numConvLayers), ...
    'perfRates', ones(1, numConvLayers), 'perfTypes', ones(1, numConvLayers), ...
    'cost', +Inf, 'loss', originalLoss, 'time', netTotalTimeOriginal);
  bestNet = [];
% end

for stepIdx = firstStepIdx:numSteps
  tic;

  bestInfo = previousBestInfo;
  bestInfo.cost = +inf;

  if all(previousBestInfo.perfIdx == length(rates))
    fprintf('Layers cannot be perforated further\n');
    break;
  end
  
  for convIdx = 1:numConvLayers
    if previousBestInfo.perfIdx(convIdx) == length(rates)
      continue;
    end
    
    curPerfTypes = previousBestInfo.perfTypes;
    curPerfIdx = previousBestInfo.perfIdx;
    curPerfIdx(convIdx) = curPerfIdx(convIdx) + 1;

    perforationTypes = convLayersData{convIdx}.perforationTypes;
    for perfTypeIdx = 1:length(perforationTypes)
      curPerfTypes(convIdx) = perfTypeIdx;
      [perfConfig, curPerfRates] = get_perf_config(curPerfIdx, curPerfTypes, rates, convLayersData);

      for rep = 1:numRepeatMask
        perfNet = perforate_all_conv_layers(net, perfConfig, convLayersData, inputSizesData, useGpu);

        perfNetTime = vl_simplenn_move(perfNet, deviceName);
        [ ~, curTime, ~ ] = net_total_time(convLayersData, perfNetTime, imdbTime, getBatchTime, batchSizeVal, numRepeat);

        valInfo = cnn_validate(perfNet, imdb, getBatch, val, batchSizeVal, useGpu, prefetch);
        curLoss = valInfo.(lossFieldName);

        if netTotalTimeOriginal <= curTime
          curCost = +Inf;
        else
          curCost = (curLoss - originalLoss) / (netTotalTimeOriginal - curTime);
        end

        fprintf('step %d conv %d %s perf time %.4f (%.2fx) perf loss %.4f (+%.4f), cost %f\n', ...
          stepIdx, convIdx, char(perforationTypes(perfTypeIdx)), ...
          curTime, netTotalTimeOriginal / curTime, curLoss, (curLoss - originalLoss), ...
          curCost);

        if curCost < bestInfo.cost
          bestInfo = struct('perfRates', curPerfRates, 'perfIdx', curPerfIdx, ...
            'perfTypes', curPerfTypes, 'time', curTime, 'loss', curLoss, 'cost', curCost);
          bestNet = perfNet;
          fprintf('Best value updated!\n');
        end
      end  
    end
  end

  fprintf('step %d results: perf time %.4f (%.2fx) perf loss %.4f (+%.4f) cost %f\n', ...
          stepIdx, bestInfo.time, netTotalTimeOriginal / bestInfo.time, ...
          bestInfo.loss, (bestInfo.loss - originalLoss), ...
          bestInfo.cost);
  perfConfig = get_perf_config(bestInfo.perfIdx, bestInfo.perfTypes, rates, convLayersData);
  for k = 1:size(perfConfig, 1)
    fprintf('%f %s\n', perfConfig{k, 1}, char(perfConfig{k, 2}));
  end

  % recalculating iterative impacts
  if any(perforationTypes == PerforationType.IterativeImpact)
    fprintf('Recalculating impacts... ');
    averageImpacts = weights_average_impact(net, imdb, getBatch, trainImpact, convLayers, batchSizeTrain);
    for convIdx = 1:numConvLayers
      convLayersData{convIdx}.iterativeImpacts{bestInfo.perfIdx(convIdx) + 1} = averageImpacts{convIdx};
    end
    fprintf('done\n');
  end

  previousBestInfo = bestInfo;
  greedyProfile{stepIdx} = bestInfo;
  
  fprintf('Saving\n');
  save([folder '.mat'], 'greedyProfile', 'convLayersData', 'inputSizesData', 'netTotalTimeOriginal', 'originalLoss');

  netTemp = net;
  net = vl_simplenn_move(bestNet, 'cpu');
  save(fullfile(folder, ['net_' num2str(stepIdx) '.mat']), 'net');
  net = netTemp;
  clear bestNet netTemp

  t = toc;
  fprintf('Time for iteration: %.2f seconds\n', t);
end

end

function [perfConfig, perfRates] = get_perf_config(perfIdx, perfTypes, rates, convLayersData)

numConvLayers = length(convLayersData);
perfConfig = cell(numConvLayers, 2);
perfRates = zeros(1, numConvLayers);
for l = 1:numConvLayers
  perfRates(l) = rates(perfIdx(l));
  perfConfig{l, 1} = perfRates(l);
  perfConfig{l, 2} = convLayersData{l}.perforationTypes(perfTypes(l));
end

end
