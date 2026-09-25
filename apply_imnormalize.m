function outarray = apply_imnormalize (in_array)
    % inarray is an NxMx3 uint8 image and outarray is NxMx3
    arguments
        in_array (:,:,:) uint8
    end

    tmp = double(in_array) / 255.0; % Normalize from [0, 255] to [0, 1]

    % Define channel means and standard deviations
    im_mean = cat(3, 0.4914, 0.4822, 0.4465);
    im_std = cat(3, 0.2470, 0.2435, 0.2616);

    % Normalize by channel using channel means and stds
    outarray = (tmp - im_mean) ./ im_std;
end