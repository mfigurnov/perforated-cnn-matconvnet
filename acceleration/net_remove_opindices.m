function [ net ] = net_remove_opindices( net )

for i = 1:numel(net.layers)
  if isfield(net.layers{i}, 'opindices')
    net.layers{i} = rmfield(net.layers{i}, 'opindices');
  end
end

end
