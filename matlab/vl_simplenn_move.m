function net = vl_simplenn_move(net, destination)
% VL_SIMPLENN_MOVE  Move a simple CNN between CPU and GPU
%    NET = VL_SIMPLENN_MOVE(NET, 'gpu') moves the network
%    on the current GPU device.
%
%    NET = VL_SIMPLENN_MOVE(NET, 'cpu') moves the network
%    on the CPU.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

switch destination
  case 'gpu', moveop = @(x) gpuArray(x) ;
  case 'cpu', moveop = @(x) gather(x) ;
  otherwise, error('Unknown desitation ''%s''.', destination) ;
end
for l=1:numel(net.layers)
  switch net.layers{l}.type
    case 'conv'
      for f = {'filters', 'biases', 'filtersMomentum', 'biasesMomentum', 'maskindices', ...
        'outindices', 'opindices'}
        f = char(f) ;
        if isfield(net.layers{l}, f)
          net.layers{l}.(f) = moveop(net.layers{l}.(f)) ;
        end
      end
      for f = {'pad', 'stride'}
        f = char(f) ;
        if isfield(net.layers{l}, f)
          net.layers{l}.(f) = double(net.layers{l}.(f)) ;
        end
      end
    case 'pool'
      if isfield(net.layers{l}, 'opindices')
        % transpose pooling opindices for GPU
        opindices = net.layers{l}.opindices;
        if strcmp(destination, 'gpu') && ~isa(opindices, 'gpuArray')
          % CPU -> GPU
          opindices = permute(opindices, [2 3 1]);
        elseif strcmp(destination, 'cpu') && isa(opindices, 'gpuArray')
          % GPU -> CPU
          opindices = permute(opindices, [3 1 2]);
        end
        net.layers{l}.opindices = moveop(opindices);
      end
      for f = {'pad', 'stride', 'pool'}
        f = char(f) ;
        if isfield(net.layers{l}, f)
          net.layers{l}.(f) = double(net.layers{l}.(f)) ;
        end
      end
    case 'perfknn'
      for f = {'maskindices', 'outindices', 'weights'}
        f = char(f) ;
        if isfield(net.layers{l}, f)
          net.layers{l}.(f) = moveop(net.layers{l}.(f)) ;
        end
      end
    otherwise
      % nothing to do ?
  end
end
