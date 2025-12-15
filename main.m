%% load the dataset
% digitDatasetPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', ...
%     'nndatasets', 'DigitDataset');
digitDatasetPath = fullfile("\train");
imds = imageDatastore(digitDatasetPath,'FileExtensions',{'.jpg','.png'},...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Count the number of images in each category
labelCount = countEachLabel(imds);

%% Divide the data into training and validation sets.
numTrainFiles =3600;
[imdsTrain, imdsValidation] = splitEachLabel(imds, numTrainFiles, 'randomize'); %splitEachLabel：拆分数据集

%% CNNframe
layers = [
    imageInputLayer([32 32 1])
    
    
    convolution2dLayer(3, 8, 'padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)     
   
    convolution2dLayer(3, 16, 'padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)     
       
    convolution2dLayer(3, 32, 'padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)  
    dropoutLayer(0.2,"Name","dropout_5")
 
    fullyConnectedLayer(512)
    reluLayer
    dropoutLayer(0.5,"Name","dropout_6")
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];

%% Options——SGDM
% trainingOptions :Training Options of Deep Learning Neural Network 
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ... 
    'MaxEpochs',100, ... 
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ... 
    'ValidationFrequency', 30, ... 
    'Verbose', false, ...
    'Plots', 'training-progress');

%% Train the network using the training set
net = trainNetwork(imdsTrain, layers, options);
 save(digitDatasetPath,'net')  %save model

%% calculate the accuracy
YPred = classify(net, imdsValidation); 
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation) / numel(YValidation); 

%%
%prediction results
figure;
nSample = 10;
ind = randperm(size(YPred,1),nSample);
for i = 1:nSample
    
subplot(2,fix((nSample+1)/2),i)
imshow(char(imdsValidation.Files(ind(i))))
title(['预测：' char(YPred(ind(i)))])
if char(YPred(ind(i))) ==char(YValidation(ind(i)))
    xlabel(['真实:' char(YValidation(ind(i)))],'Color','b')
else
    xlabel(['真实:' char(YValidation(ind(i)))],'color','r')
end 

end
