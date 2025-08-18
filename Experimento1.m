clc
clear
close all



%load dataset
load ionosphere.mat

rng(10);
%Convert Y to numeric classes
[Y, clasesNames] = grp2idx(Y);

%Train test datasets 80 20
cv = cvpartition(size(X,1),"HoldOut", 0.2);
idx = cv.test;

Xtrain = X(~idx,:);
Ytrain = Y(~idx,:);
Xtest = X(idx,:);
Ytest = Y(idx,:);

%fit forest
forest = FitRandForest(Xtrain, Ytrain);

% Accuracy over the same training dataset
accuracy = 100*sum(PredTreeBagging(forest,Xtest) == Ytest) / length(Ytest);
disp(accuracy);



% MATLAB COMPARISON 
%Mdl = TreeBagger(NumTrees,X,Y) returns Mdl trained by the predictor data in the matrix X and the class labels in the array Y.

treebag = TreeBagger(100, Xtrain, Ytrain);
% Yfit = predict(B,X)

accuracy_bag = 100*sum(str2double(predict(treebag,Xtest)) == Ytest) / length(Ytest);

disp(accuracy_bag);









% %%%%%%%%%%%    2nd part: try parameters combinations%%%%%%%%%
% 
% %Create parameter configurations
% nTreesList = [10,20,30];
% nSamplesList = [20,30,50,60];
% %Create grid to cover all the combinations
% [T,S] = ndgrid(nTreesList, nSamplesList);
% combinations = [T(:),S(:)];
% 
% results=[];
% 
% for i = 1:size(combinations,1)
%     nTrees = combinations(i,1);
%     nSamples = combinations(i,2);
% 
%     fprintf("Trying with nTrees = %d, nSamples = %d ... \n", nTrees, nSamples);
%     model = RandomForestBagging(X,Y, nTrees, nSamples);
%     model.train(varNames);
%     prediction = model.predict(X);
%     acc = 100*sum(prediction == Y)/length(Y);
%     results = [results; nTrees, nSamples, acc];
% end
% 
% disp('nTrees | nSamples | Accuracy');
% disp(results);
% 
% 
% %Visualización resultado
% AccMatrix = reshape(results(:,3), length(nSamplesList), length(nTreesList));
% figure;
% surf(nTreesList, nSamplesList, AccMatrix);
% xlabel('nTrees');
% ylabel('nSamples');
% zlabel('Accuracy (%)');
% title('Accuracy vs nTrees y nSamples');
% colorbar;