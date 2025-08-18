function predictions = Conformal_Prediction_Predict(forest,qhat, Xtest, nclasses)
%CONFORMAL_PREDICTION_PREDICT Given a set of bagged decision trees and the
%value of the boundary for conformal predictions qhat
%produce for each instance a set of labels with probability higher than
%1-qhat
%nclasses is the number of possible classes

    %compute the average probabilities for our bagged trees
    scores = zeros(size(Xtest,1), nclasses);
    
    %avergaing system
    % for i = 1:length(forest)
    %     tree = forest{i};
    %     [~, prediction_tree] = predict(tree, Xtest);
    %     scores = scores + prediction_tree;
    % end
    % scores = scores / length(forest); % take the average of the probabilities

    %voting system
    for i = 1:length(forest)
        tree = forest{i};
        predictions_tree = predict(tree, Xtest);
        for j = 1:size(Xtest,1)
            c = predictions_tree(j);
            scores(j,c) = scores(j,c)+1;
        end
    end

    scores = scores / length(forest);

    boundary = 1 - qhat;
    n = size(Xtest, 1);
    predictions = cell(n, 1); % array to store the sets of predictions
    
    for i = 1:n
        % return classes with probability higher thatn the boundary
        predictions{i} = find(scores(i, :) >= boundary);
    end
end

