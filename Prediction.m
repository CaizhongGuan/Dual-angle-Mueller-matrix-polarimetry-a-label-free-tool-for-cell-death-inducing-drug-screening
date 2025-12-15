clc
clear all
close all

%% load the model

%% prediction
digitDatasetPath = fullfile("\prediction");
imds = imageDatastore(digitDatasetPath,'FileExtensions',{'.jpg','.png'},...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
Pred = classify(net, imds);
Lab=imds.Labels;
accuracy = sum(Pred == Lab) / length(Lab) %caculate the accuracy
fprintf('Accuracy is %f\n', accuracy);
confusion_matrixfenlei2(Lab,Pred); 