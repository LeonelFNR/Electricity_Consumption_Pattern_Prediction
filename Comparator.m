function Comparator(X,Y, nTrees)
    %COMPARATOR Compares the performance of my own TreeBagged implementation
    %and MATLAB's own implementation
    %X and Y must have numbers

    if nargin < 3
        nTrees = 100;
    end

    %it divides the dataset in 80% for training and 20% for testing
    %Train test datasets 80 20
    rng(10); %seed
    cv = cvpartition(size(X,1),"HoldOut", 0.2);
    idx_test = cv.test;
    idx_train = cv.training;
    
    Xtrain = X(idx_train,:);
    Ytrain = Y(idx_train,:);
    Xtest = X(idx_test,:);
    Ytest = Y(idx_test,:);
    
    %fit forest
    forestPers = FitTreeBagging(Xtrain, Ytrain, nTrees);
    
    % Accuracy over the same training dataset
    accuracyPersImpl = sum(PredTreeBagging(forestPers,Xtest) == Ytest) / length(Ytest);
    fprintf('The obtained accuracy for my personal implementation is %.2f%%\n', accuracyPersImpl*100);    
    
    % MATLAB COMPARISON 
    %Mdl = TreeBagger(NumTrees,X,Y) returns Mdl trained by the predictor data in the matrix X and the class labels in the array Y.
   
    forestMAT = TreeBagger(nTrees, Xtrain, Ytrain);   
    accuracyMatImpl = sum(str2double(predict(forestMAT,Xtest)) == Ytest) / length(Ytest);
    fprintf('The obtained accuracy for MATLAB implementation is %.2f%%\n', accuracyMatImpl*100); 
end

