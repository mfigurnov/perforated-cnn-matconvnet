% Load data
net_vgg=load('imagenet-vgg-verydeep-16-border.mat');
images_vgg.data=load('im3000_train_vgg.mat');
images_vgg.data=images_vgg.data.im3000_vgg;

%Given parameters are provided for VGG, they are adjustable to any trained CNN
%Calculate individual decomposition layer sizes 
conv_l_vec=[3 6 8 11 13 15 18 20 22 25 27 29]; 
%Leave conv1 for vgg unchanged, accelerate from No.3 because of complexity reasons
d_max=512;
speedup_ratio=4;
batchSize=8;
batchNum=40;
d__vec = get_rank_vector(speedup_ratio, net_vgg, images_vgg.data, batchNum, batchSize, conv_l_vec, d_max, true);

%Speedup net
batchSize=8;
[net_new4x] = speedup_net_linear(net_vgg, net_vgg, batchNum, batchSize, images_vgg, conv_l_vec, d__vec, false, true);