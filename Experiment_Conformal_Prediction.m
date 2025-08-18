clear all
close all

dataset = readtable("dry_bean_dataset.csv");
X = dataset(:,1:end-1); %copy all columns minus the last, because it has the labels
Y = dataset(:,end);

rng(13);
[Y,] = grp2idx(dataset{:,end});
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

tree = fitctree(Xtrain, Ytrain);
[~, scores] = predict(tree, Xval);
s_scores = [];
for i=1:size(Yval,1)
    val_class = Yval(i);
    s_score_val = 1 - scores(i, val_class);
    s_scores = [s_scores; s_score_val];
end

n = size(Xval,1);
alpha = 0.1;
q_quantile = ceil((n+1)*(1-alpha))/n;
qhat = quantile(s_scores, min(1,q_quantile)); %compute quantile with a safety check

%generate predictions SETS for test set
[~,predictions_test] = predict(tree, Xtest);
boundary = 1 - qhat;

predictions_test(predictions_test < boundary) = 0;

for i = 1:size(predictions_test,1)
    predictions_test(i,:) = normalize(predictions_test(i,:), 'norm', 1);
end

