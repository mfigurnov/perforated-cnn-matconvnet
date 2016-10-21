function net_new = net_rm_indices(net_new, layer_no) 
    % Removes perforation indices od the CNN if necessary

    if isfield(net_new.layers{layer_no+1}, 'nonPerforatedIndices')
        net_new.layers{layer_no+1} = ...
            rmfield(net_new.layers{layer_no+1}, 'nonPerforatedIndices');
    end
    
    if isfield(net_new.layers{layer_no+1}, 'interpolationIndicesOut')
        net_new.layers{layer_no+1} = ...
            rmfield(net_new.layers{layer_no+1}, 'interpolationIndicesOut');
    end
    
    if isfield(net_new.layers{layer_no+1}, 'interpolationIndicesIn')
        net_new.layers{layer_no+1} = ...
            rmfield(net_new.layers{layer_no+1}, 'interpolationIndicesIn');
    end
    
    if isfield(net_new.layers{layer_no+1}, 'rate')
        net_new.layers{layer_no+1} = ...
            rmfield(net_new.layers{layer_no+1}, 'rate');
    end
    
    if isfield(net_new.layers{layer_no+1}, 'opindices')
        net_new.layers{layer_no+1} = ...
            rmfield(net_new.layers{layer_no+1}, 'opindices');
    end
end