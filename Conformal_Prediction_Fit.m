function qhat = Conformal_Prediction_Fit(forest,X_cal, Y_cal, alpha, nclasses)
%CONFORMAL_PREDICTION_FIT 
%Given a trained set of bagged trees (forest) and calibration data (X_cal, Y_cal), this function
%computes the boundary qhat for which a 
%prediction set can be provided for making predictions with certainty 1 -
%alpha, where alpha is a spcefied error
%nclasses is the number of possible classes

    %compute the average scores for our bagged trees
    scores = zeros(size(size(X_cal,1),nclasses));
    
    for i = 1:length(forest)
        tree = forest{i};
        [~, prediction_tree] = predict(tree, X_cal);
        scores = scores + prediction_tree;
    end

    scores = scores / length(forest); % take the average of the probabilities

    s_scores = [];
    %compute the s_scores
    for i=1:size(Y_cal,1)
        cal_class = Y_cal(i);
        s_score_val = 1 - scores(i, cal_class);
        s_scores = [s_scores; s_score_val];
    end
    
    n = size(X_cal,1);
    q_quantile = ceil((n+1)*(1-alpha))/n;
    qhat = quantile(s_scores, min(1,q_quantile)); %compute quantile with a safety check

end

