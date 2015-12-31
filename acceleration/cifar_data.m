function [net, imdb, getBatch, train, val, getBatchTrain] = cifar_data(useGpu, varargin)

opts.dataDir = '';
opts.expDir = '';
opts.modelPath = fullfile(vl_rootnn, 'cifar10-nin.mat');
opts.imdbPath = fullfile(vl_rootnn, 'cifar10-imdb.mat');
opts = vl_argparse(opts, varargin) ;

getBatch = @getBatchFunc;
getBatchTrain = getBatch;

imdb = load(opts.imdbPath) ;
if useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end

train = find(imdb.images.set==1);
val = find(imdb.images.set==3);

net = net_load(useGpu, opts.modelPath);

% Switch the network to implementation with cached convolution and pooling indices
inputSizesData = net_input_sizes(net, imdb, getBatch, 2, useGpu);
net = net_set_opindices(net, inputSizesData, useGpu);

end

function [im, labels] = getBatchFunc(imdb, batch)
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
end
