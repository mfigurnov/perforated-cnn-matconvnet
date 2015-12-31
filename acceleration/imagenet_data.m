function [net, imdb, getBatch, train, val, getBatchTrain] = imagenet_data(useGpu, varargin)

opts.lite = false;
opts.numFetchThreads = 2;
opts.dataDir = '/mnt/ssd/imagenet12';
opts.expDir = fullfile(vl_rootnn,'acceleration','data','alexnet');
opts.imdbPath = fullfile(vl_rootnn,'acceleration','data','alexnet','imdb.mat');
opts.modelPath = fullfile(vl_rootnn, 'imagenet-caffe-ref.mat');
opts = vl_argparse(opts, varargin);

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath);
else
  imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite);
  if ~exist(opts.expDir)
    mkdir(opts.expDir);
  end
  save(opts.imdbPath, '-struct', 'imdb') ;
end

train = find(imdb.images.set==1);
val = find(imdb.images.set==2);

net = net_load(useGpu, opts.modelPath);

% Very important fix for VGG16 :-| (11.6% -> 10.1% top-5 error)
% See https://github.com/vlfeat/matconvnet/issues/296
% The line is commented out since this setting is included in
% imagenet-vgg-verydeep-16-border.mat file
% net.normalization.border = [256 256] - net.normalization.imageSize(1:2) ;

getBatch = getBatchWrapper(net.normalization, opts.numFetchThreads, 'none', useGpu);
getBatchTrain = getBatchWrapper(net.normalization, opts.numFetchThreads, 'f25', useGpu);

% Switch the network to implementation with cached convolution and pooling indices
inputSizesData = net_input_sizes(net, imdb, getBatch, 2, useGpu);
net = net_set_opindices(net, inputSizesData, useGpu);

end

function fn = getBatchWrapper(opts, numThreads, augmentation, useGpu)
fn = @(imdb,batch) getBatch(imdb,batch,opts,numThreads, augmentation, useGpu);
end

function [im,labels] = getBatch(imdb, batch, opts, numThreads, augmentation, useGpu)
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = cnn_imagenet_get_batch(images, opts, ...
                            'numThreads', numThreads, ...
                            'prefetch', nargout == 0, ...
                            'augmentation', augmentation);
if nargout ~= 0 && useGpu
  im = gpuArray(im);
end
labels = imdb.images.label(batch) ;
end
