function Ytestpred = PredRandForest(forest,Xtest)
%PREDRANDFOREST given a forest and input dataset it predicts the output
%   via the modal result of the predictions of a set of decicision trees in
%   forest
% Xtest is the dataset
% forest is the collection of trees. THEY MUST HAVE BEEN PREVIOUSLY TRAINEd
    nTrees = numel(forest);
    nTest = size(Xtest, 1);
    all_preds = zeros(nTest, nTrees); % matriz numérica

    for i = 1:nTrees
        tree = forest{i};
        pred = predict(tree, Xtest);
        all_preds(:, i) = pred;
    end

    % votación por la moda
    Ytestpred = mode(all_preds, 2);
end

