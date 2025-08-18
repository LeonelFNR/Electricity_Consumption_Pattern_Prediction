clear all
close all

rng(10);

data = load("descarga_dataset_edificios\Energy-Datasets\Apartment-Dataset\Dataset_Consumption_Prediction_48hours_Week&Day_Before_classEncoded_Normalized.mat");
fieldNames = fieldnames(data);
dataset = data.(fieldNames{1});


%data preparation
X = dataset(:,1:end-1);
Y = dataset(:,end);
%[Y,] = grp2idx(second_dataset{:,end}); % uncomment if using dataset 2
nclasses = numel(unique(Y));
cvtraintest = cvpartition(size(X,1), "Holdout", 0.2);
idx = cvtraintest.test;
idx_trainval = cvtraintest.training;
Xtrainval = X(idx_trainval, :);
Ytrainval = Y(idx_trainval, :);
Xtest = X(idx, :);
Ytest = Y(idx, :);
cvtrainval = cvpartition(size(Xtrainval,1), "Holdout", 0.25);
idx_val = cvtrainval.test;
idx_train = cvtrainval.training;
Xval = X(idx_val, :);
Xval = Xtrainval(idx_val,:);
Yval = Ytrainval(idx_val, :);
Xtrain = Xtrainval(idx_train,:);
Ytrain = Ytrainval(idx_train,:);

%fit tree bagging
forest = FitTreeBagging(Xtrain, Ytrain, 200);

%prediction and assesment
predictions_def = PredTreeBagging(forest, Xtest);
accuracy = sum(Ytest == predictions_def) / length(Ytest);
fprintf("The accuracy of the tree bagger implementation is: %.3f \n", accuracy);

%conformal prediction
alpha = 0.1; % error we will allow
qhat = Conformal_Prediction_Fit(forest, Xval, Yval, alpha, nclasses);
predictions_conf = Conformal_Prediction_Predict(forest, qhat, Xtest, nclasses);

%assesment conformal prediction
test_vector = [];
for i = 1:length(Ytest)
    test_vector = [test_vector; ismember(Ytest(i), predictions_conf{i})];
end

accuracy_conf = sum(test_vector)/length(test_vector);
fprintf("The accuracy of the conformal prediction implementation is: %.3f \n", accuracy_conf);

%compute the average set size when doing conformal prediction
n_elem = 0;
for i = 1:length(predictions_conf)
n_elem = n_elem + numel(predictions_conf{i});
end
n_avg = n_elem / length(predictions_conf);

fprintf("The average set size for the conformal prediction is: %.3f \n", n_avg);

