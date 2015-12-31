function vgg_greedy_perforation(varargin)

opts.dataDir = '/home/mfigurnov/imagenet12-aspect';
opts.expDir = fullfile(vl_rootnn,'acceleration','data','vgg16-grid');
opts.imdbPath = fullfile(vl_rootnn,'acceleration','data','vgg16-grid','imdb.mat');
opts.modelPath = fullfile(vl_rootnn, 'imagenet-vgg-verydeep-16-border.mat');

opts.batchSizeTrain = 8;
opts.batchSizeVal = 16;
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
opts.numSteps = 150;
opts.numRepeat = 5;
opts.numRepeatMask = 2;
opts.perforationTypes = PerforationType.Grid; % PerforationType.IterativeImpact;
opts = vl_argparse(opts, varargin);

net_greedy_perforation(@imagenet_data, opts.rates, opts.dataDir, opts.expDir, opts.imdbPath, ...
  opts.modelPath, opts.batchSizeTrain, opts.batchSizeVal, opts.useGpu, opts.useGpuTimings, opts.prefetch, ...
  opts.lossFieldName, opts.numSteps, opts.trainImpactSize, opts.trainSize, opts.valSize, ...
  opts.validateOnTrain, opts.numRepeat, opts.numRepeatMask, opts.perforationTypes);

end
