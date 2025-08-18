function forest = FitRandForest(X,Y, nTrees)
%TRAINRANDFOREST Summary of this function goes here
% Performs the training of a set of trees
% X is the variables dataset
% Y is the output dataset

  if nargin < 3
    nTrees = 100;
  end
  forest = cell(1,nTrees);
  nSamples= size(X, 1);


  % playing with fitctree
    nFeatures = size(X, 2);
    numVarsToSample = round(sqrt(nFeatures));
    minLeafSize = 1;

  for i = 1:nTrees
      %Select random indexes with substitution
      idx = randsample(nSamples, nSamples, true);

      X_sub = X(idx, :);
      Y_sub = Y(idx);

      % Train each tree with subdataset
      %tree = fitctree(X_sub, Y_sub);

      %playing with fitctree
      tree = fitctree(X_sub, Y_sub, ...
            'NumVariablesToSample', numVarsToSample, ...
            'MinLeafSize', minLeafSize, ...
            'SplitCriterion', 'gdi');  % o 'deviance'

      forest{i} = tree;
  end
end

