function net_new = create_new_net(net_old, W1, W2, b1, b2, layer_no)
% Creates a new net by the given old one, weights and biases of the updated net
%layer_no is the number of the conv layer which is going to be decomposed

    % new layers
    first_layer=net_old.layers{layer_no};
    first_layer.filters=W1;
    first_layer.biases=b1;
    second_layer=net_old.layers{layer_no};
    second_layer.filters=W2;
    second_layer.biases=b2;

    % new net
    net_new = net_old;
    net_new.layers(layer_no+2:end+1)=net_old.layers(layer_no+1:end);
    net_new.layers{layer_no}=first_layer;
    net_new.layers{layer_no+1}=second_layer;
    net_new.layers{layer_no+1}.stride = [1 1];
    net_new.layers{layer_no+1}.pad= [0 0 0 0];
end