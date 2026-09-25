function outarray = apply_maxpool(inarray)
    %inarray is 2Nx2MxD and outarray is size NxMxD

    % Compute number of output columns and rows and then create outputarray
    % of this size
    out_rows = size(inarray, 1) / 2;
    out_cols = size(inarray, 2) / 2;
    outarray = zeros(out_rows, out_cols, size(inarray, 3));

    % iterate through rows and columns with stride 2
    for row = 1:2:size(inarray, 1)
        for col = 1:2:size(inarray, 2)
            % Extract the 2-by-2 pooling window
            window = inarray(row:row+1, col:col+1, :);

            % Compute output row and column corresponding to pooling window
            out_row = (row-1) / 2 + 1;
            out_col = (col-1) / 2 + 1;

            % Find maximum value in window and set value in outarray
            outarray(out_row, out_col, :) = max(window, [], [1 2]);
        end
    end
end