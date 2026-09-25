function outarray = apply_fullconnect(inarray, filterbank, biasvals)
    %inarray is NxMxD1, filterbank is NxMxD1xD2,
    %biasvals is a length D2 vector, and outarray is 1x1xD2

    [N, M, D1, D2] = size(filterbank);

    % initialize output array
    outarray = zeros(1, 1, D2);

    % Compute the sum of F_l(i,j,k)×In(i,j,k) and add bias
    for l = 1:D2
        outarray(1, 1, l) = sum(inarray .* filterbank(:, :, :, l), 'all') + biasvals(l);
    end
end