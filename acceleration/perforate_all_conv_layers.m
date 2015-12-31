function [ net ] = perforate_all_conv_layers(net, perfConfig, convLayersData, inputSizesData, useGpu)

for i = 1:length(convLayersData)
  % Skip layers with rate = 1 (non-perforated)
  if perfConfig{i, 1} ~= 1
    net = perforate_conv_layer(net, perfConfig{i, 1}, perfConfig{i, 2}, convLayersData{i});
  end
end

% Calculate indices for convolutions and poolings. This activates perforation.
net = net_set_opindices(net, inputSizesData, useGpu);

end
