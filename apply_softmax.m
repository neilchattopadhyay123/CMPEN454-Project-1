function outarray = apply_softmax(inarray)
    %inarray is 1x1xD and outarray is the same size
    outarray = zeros(size(inarray))

    alpha = max(inarray(1,1,:));
    sumexp = sum(exp(inarray(1,1,:) - alpha), 'all');
    for k = 1:size(outarray, 3)
        outarray(1,1,k) = exp(inarray(1,1,k) - alpha) / sumexp;
    end
end