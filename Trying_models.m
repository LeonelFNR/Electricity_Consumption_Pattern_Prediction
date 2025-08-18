clear all
close all
clc

load("descarga_dataset_edificios\Energy-Datasets\Apartment-Dataset\Dataset_Consumption_Prediction_48hours_Week&Day_Before_classEncoded_Normalized.mat")
X = dataset(:,1:end-1);
Y = dataset(:,end);

rng(10); % seed for random numbers

cv = cvpartition(size(X,1), "HoldOut", 0.2);
Xtrain = X(cv.training, :);
Ytrain = Y(cv.training);
Xtest = X(cv.test, :);
Ytest = Y(cv.test);

% Use the personal implementation of the tree bagging
forest = FitTreeBagging(Xtrain, Ytrain);
predictions_forest = PredTreeBagging(forest, Xtest);
accuracy_forest = sum(predictions_forest == Ytest) / length(Ytest);

%Apply fitcauto
%Mdl = fitcauto(Xtrain, Ytrain, 'HyperparameterOptimizationOptions', struct('Optimizer', 'asha'));
Mdl = fitcauto(Xtrain, Ytrain);
save("bestFoundModel.mat", "Mdl");

predictions_model = Mdl.predict(Xtest);
accuracy_model = sum(predictions_model == Ytest) / length(Ytest);