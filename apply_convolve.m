function outarray = apply_convolve(inarray, filterbank, biasvals)
    %inarray is NxMxD1, filterbank is RxCxD1xD2,
    %biasvals is a length D2 vector, and outarray is NxMxD2

    % Input dimensions
    [N, M] = size(inarray, [1, 2]);
    [R, C, D1, D2] = size(filterbank);


    outarray = zeros(N, M, D2);

    for l = 1:D2
        temp = zeros(N, M); % intial NxM array for output channel l

        % Compute F_l(:,:,k) ∗ In(:,:,k) and add it to the output array
        for k = 1:D1
            temp = temp + imfilter(inarray(:,:,k), filterbank(:,:,k,l), 'same', 0, 'conv'); 
        end
    
        % Add bias values to output array
        outarray(:,:,l) = temp + biasvals(l); 
    end
end