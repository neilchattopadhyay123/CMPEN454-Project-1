function outarray = apply_relu(inarray)
    %inarray is NxMxD and outarray is the same size
    
    outarray = max(inarray, 0); % apply ReLU Definition
end