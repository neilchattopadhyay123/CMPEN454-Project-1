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

fprintf('\nCNN Debugging Test\n');

% CNN parameters
computed = imrgb;

for d = 1:length(layertypes) 
    % current layer type and CNN parameters
    layer = layertypes{d};
    filterbank = filterbanks{d};
    biasvec = biasvectors{d};

    % Apply CNN operations based on what it is
    if strcmp(layer, 'imnormalize')
        computed = apply_imnormalize(computed);
    
    elseif strcmp(layer, 'convolve')
        computed = apply_convolve(computed, filterbank, biasvec);
    
    elseif strcmp(layer, 'relu')
        computed = apply_relu(computed);
    
    elseif strcmp(layer, 'maxpool')
        computed = apply_maxpool(computed);
    
    elseif strcmp(layer, 'fullconnect')
        computed = apply_fullconnect(computed, filterbank, biasvec);

    elseif strcmp(layer, 'softmax')
        computed = apply_softmax(computed);

    end
    
    computedLayer{d} = computed;
    expected = layerResults{d};
    totalDiff = max(abs(computed(:) - expected(:)));

    % print out all info 
    fprintf('Layer %2d (%-4s)', d, layer);
    fprintf(' size = %d x %d x %d', size(computed, 1), size(computed, 2), size(computed, 3));
    fprintf(' max difference = %.4g\n', totalDiff);
end

% find expected most probable class 
expected_classprobvec = squeeze(layerResults{end});
[expected_maxprob, expected_maxclass] = max(expected_classprobvec);

% note, classlabels is defined in 'cifar10testdata.mat'
fprintf('expected estimated class is %s with probability %.4f\n', classlabels{expected_maxclass}, expected_maxprob);

% find computed most probable class 
computed_classprobvec = squeeze(computed);
[computed_maxprob, computed_maxclass] = max(computed_classprobvec);

% note, classlabels is defined in 'cifar10testdata.mat'
fprintf('computed estimated class is %s with probability %.4f\n', classlabels{computed_maxclass}, computed_maxprob);


% Displaying debugging image 
figure; 
imagesc(imrgb);
axis image;
axis off;
title(sprintf('Debugging Image'))

% Displaying Intermediate image
% gemini suggest code --> displayed intermediate pretty well
% tweaked code a bit --> kept same idea as suggested code
layerIntResults = computedLayer{2};
figure;
for c = 1:size(layerIntResults, 3)
    subplot(2, 5, c);
    imagesc(layerIntResults(:, :, c));
    axis image;
    axis off;
    title(sprintf('Channel %d', c));
end
colormap gray;

% Displaying softmax probability
% gemini suggested code --> displayed softmax good enough so I kept it
% tweaked the code a bit --> overall same idea as suggested code
figure;
bar(computed_classprobvec);
set(gca, 'XTick', 1:numel(classlabels), 'XTickLabel', classlabels);
xtickangle(45);
xlabel('Image');
ylabel('Probability');
title('CNN Probabilities');

% Running all CIFAR-10 Test
% rows = true class
% columns = predicted class
fprintf('\nCIFAR-10 Performance\n');

% confusion matrix for later
C = zeros(10,10);

% go through all images in CIFAR-10
for i = 1: size(imageset, 4)

    % iterate through all 18 layers --> keep track of output array at each one 
    layerResults = cell(1, length(layertypes));

    for j = 1: length(layertypes)
        layer = layertypes{j};
        filterbank = filterbanks{j};
        biasvec = biasvectors{j};

        if strcmp(layer, 'imnormalize')
            layerResults{j} = apply_imnormalize(imageset(:, :, :, i));
        elseif strcmp(layer, 'convolve')
            layerResults{j} = apply_convolve(layerResults{j-1}, filterbank, biasvec);
        elseif strcmp(layer, 'relu')
            layerResults{j} = apply_relu(layerResults{j-1});
        elseif strcmp(layer, 'maxpool')
            layerResults{j} = apply_maxpool(layerResults{j-1});
        elseif strcmp(layer, 'fullconnect')
            layerResults{j} = apply_fullconnect(layerResults{j-1}, filterbank, biasvec);
        elseif strcmp(layer, 'softmax')
            layerResults{j} = apply_softmax(layerResults{j-1});
        end
    end


    % Find final softmax probabilities 
    classprobvec = squeeze(layerResults{end});

    % Find predicted class
    [~, predictClass] = max(classprobvec);
    trueClass = trueclass(i);

    % confusion matrix 
    C(trueClass, predictClass) = C(trueClass, predictClass) + 1;

    if mod(i, 1000) == 0
        fprintf('Processed %d/%d images\n', i, size(imageset, 4));
    end

end

% accuracy 
acc = trace(C) / sum(C(:));
fprintf('\nConfusion Matrix: \n')
disp(C);

fprintf('CNN accuracy is %.2f%%\n', acc * 100);







