function net = net_load(useGpu, modelPath)

net = load(modelPath);
if isfield(net, 'net')
  net = net.net;
end
if isfield(net, 'bestNet')
  net = net.bestNet;
end
net.layers{end}.type = 'softmaxloss';

if useGpu
  net = vl_simplenn_move(net, 'gpu');
end

end

