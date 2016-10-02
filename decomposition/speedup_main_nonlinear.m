% Load data
net_vgg=load('imagenet-vgg-verydeep-16-border.mat');
images_vgg.data=load('im3000_train_vgg.mat');
images_vgg.data=images_vgg.data.im3000_vgg;

%Given parameters are provided for VGG, they are adjustable to any trained CNN
%Calculate individual decomposition layer sizes 
conv_l_vec=[3 6 8 11 13 15 18 20 22 25 27 29]; %Leave conv1 for vgg unchanged, starting from No.3
%Because of complexity reasons
d_max=512;
speedup_ratio=4;
batchSize=8;
batchNum=40;
d__vec = get_rank_vector(speedup_ratio, net_vgg, images_vgg.data, batchNum, batchSize, conv_l_vec, d_max, true);

%Speedup net
batchSize=8;
per_layer_batch_num = batchNum*ones(1,length(conv_l_vec));
lambda_vec=[ones(1,5)*0.01 ones(1,5)*1];
[net_new4x] = speedup_net_nonlinear(net_vgg, net_vgg, per_layer_batch_num, batchSize, images_vgg, 0, 0, conv_l_vec, d__vec, lambda_vec, true);
