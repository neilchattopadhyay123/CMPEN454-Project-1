% Main Program --> debugging and evaluating CNN 
%
% Loads CIFAR-10 dataset and CNN parameters
% Displays sample image from each class
% Displays info of CNN layers
% Runs debugging through all 18 layers
% compare results (ours vs known result)
% Display intermediate/final results
% run 10,000 CIFAR-10 test images
% Computes confusion matrix and accuracy 


% loads CIFAR-10 dataset 
% loading this file defines 
%   imageset (10,000 CIFAR-10 test image)
%   trueclass (correct class for each image)
%   classlabels (names of the 10 classes)
load 'cifar10testdata.mat'

% some sample code to read and display one image from each class
for classindex = 1:10
    %get indices of all images of that class
    inds = find(trueclass==classindex);

    %take first one
    imrgb = imageset(:,:,:,inds(1));

    %display it along with ground truth text label
    figure; imagesc(imrgb); truesize(gcf,[64 64]);
    title(sprintf('\%s',classlabels{classindex}));
end

% loads CNN parameters 
% loading this file defines 
%   filterbanks (filters used in convolution/fullconnect layers)
%   biasvectors (bias value in convolution/fullconnect layers)
load 'CNNparameters.mat'

fprintf("CNN Info Layer\n");
% sample code to verify which layers have filters and biases
for d = 1:length(layertypes)
    fprintf('layer %d is of type %s\n',d,layertypes{d});
    filterbank = filterbanks{d};

    if not(isempty(filterbank))
        fprintf(' filterbank size %d x %d x %d x %d\n', ...
            size(filterbank,1),size(filterbank,2), ...
            size(filterbank,3),size(filterbank,4));

        biasvec = biasvectors{d};
        fprintf(' number of biases is %d\n',length(biasvec));
    end
end


% DebuggingTest
%   imrgb        --> contains a sample test image 
%   layerResults --> Expected output array of CNN 
% Compare output at every layer with provided correct results for CNN 
% loading debuggingTest
load 'debuggingTest.mat'

fprintf('CNN Debugging Test\n');

% CNN parameters
computed = imrgb;

for d = 1:length(layertypes) 
    layer = layertypes{d};
    filterbank = filterbanks{d};
    biasvec = biasvectors{d};

    % Apply CNN operations based on what it is
    if strcmp(layer, 'imnormalize')
        computed = apply_imnormalize(computed);
    end

    if strcmp(layer, 'convolve')
        computed = apply_convolve(computed, filterbank, biasvec);
    end

    if strcmp(layer, 'relu')
        computed = apply_relu(computed);
    end

    if strcmp(layer, 'maxpool')
        computed = apply_maxpool(computed);
    end

    if strcmp(layer, 'fullconnect')
        computed = apply_fullconnect(computed, filterbank, biasvec);
    end

    if strcmp(layer, 'softmax')
        computed = apply_softmax(computed);
    end
    
    expected = layerResults{d};
    totalDiff = max(abs(computed(:) - expected(:)));

    % print out all info 
    fprintf('Layer %2d (%-12s)', d, layer);
    fprintf(' size = %d x %d x %d', size(computed, 1), size(computed, 2), size(computed, 3));
    fprintf(' max difference = %.12g\n', totalDiff);
end

% find expected most probable class 
expected_classprobvec = squeeze(layerResults{end});
[expected_maxprob, expected_maxclass] = max(expected_classprobvec);

% note, classlabels is defined in 'cifar10testdata.mat'
fprintf('expected estimated class is %s with probability %.4f\n', classlabels{expected_maxclass}, expected_maxprob);

% find computed most probable class 
computed_classprobvec = squeeze(layerResults{end});
[computed_maxprob, computed_maxclass] = max(computed_classprobvec);

% note, classlabels is defined in 'cifar10testdata.mat'
fprintf('computed estimated class is %s with probability %.4f\n', classlabels{computed_maxclass}, computed_maxprob);
