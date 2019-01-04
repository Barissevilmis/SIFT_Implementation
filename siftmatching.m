I = imread('dataset/roofs1.jpg');
I_comp = imread('dataset/roofs2.jpg');

% feature1 = load('feature1.mat');
% features1 = feature1.feature_vec;
% validpoints1 = load('validpoints1.mat');
% vpts1 = validpoints1.validpoints;

[features1, vpts1] = siftfeature(I);
[features2, vpts2] = siftfeature(I_comp);

% feature2 = load('feature2.mat');
% features2 = feature2.feature_vec;
% validpoints2 = load('validpoints2.mat');
% vpts2 = validpoints2.validpoints;

[indexPairs,matchmetric] = matchFeatures(features1,features2,'Unique',true);%,'MaxRatio',0.2);

matchedLoc1 = vpts1(indexPairs(:,1),:);
matchedLoc2 = vpts2(indexPairs(:,2),:);

figure;showMatchedFeatures(I,I_comp,matchedLoc1,matchedLoc2,'montage');