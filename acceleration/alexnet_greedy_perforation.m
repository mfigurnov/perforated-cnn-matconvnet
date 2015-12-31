function alexnet_greedy_perforation(varargin)

opts.dataDir = '/mnt/ssd/imagenet12';
opts.expDir = fullfile(vl_rootnn,'acceleration','data','alexnet');
opts.imdbPath = fullfile(vl_rootnn,'acceleration','data','alexnet','imdb.mat');
opts.modelPath = fullfile(vl_rootnn, 'imagenet-caffe-ref.mat');

opts.batchSizeTrain = 128;
opts.batchSizeVal = 256;
opts.useGpu = true;
opts.useGpuTimings = false;
opts.trainImpactSize = 5000;
opts.trainSize = 0;
opts.valSize = 2000;
opts.validateOnTrain = true;
opts.rates = [1 1/1.5 1./(2:20)];
% opts.rates = [1 1/1.5 1/2 1/3 1/4 1/5 1/6 1/7 1/8 1/9 1/10];
opts.lossFieldName = 'objective';
opts.prefetch = true;
opts.numSteps = 70;
opts.numRepeat = 5;
opts.numRepeatMask = 2;
opts.perforationTypes = PerforationType.IterativeImpact;
opts = vl_argparse(opts, varargin);

net_greedy_perforation(@imagenet_data, opts.rates, opts.dataDir, opts.expDir, opts.imdbPath, ...
  opts.modelPath, opts.batchSizeTrain, opts.batchSizeVal, opts.useGpu, opts.useGpuTimings, opts.prefetch, ...
  opts.lossFieldName, opts.numSteps, opts.trainImpactSize, opts.trainSize, opts.valSize, ...
  opts.validateOnTrain, opts.numRepeat, opts.numRepeatMask, opts.perforationTypes);

end
