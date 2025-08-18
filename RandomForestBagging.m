% Attempt of creation of random forest with only bagging
classdef RandomForestBagging < handle % Handle for easier attribute updating
    properties
        X           %Input data
        Y           %Labels
        nTrees      %Number of Trees
        trees       %Container of the trees (forest)
        nSamples    %Number of Samples that each forest has
    end

    methods
        %Builder
        function obj = RandomForestBagging(X,Y,nTrees,nSamples)
            obj.X = X;
            obj.Y = Y;
            obj.nTrees = nTrees;
            obj.nSamples = nSamples;
            obj.trees = cell(1,nTrees); %Trees storage space
        end
        
        %Function to train the trees
        function obj = train(obj, varNames) %varNames is a vector with the variables/features names
            %check that varNames and X have the same number of components
            if size(obj.X, 2) ~= length(varNames)
                error("varNames ha de tener el mismo número de columnas que X: %d vs. %d", ...
                    length(varNames), size(obj.X,2));
            end

            %Train trees through bagging: sampling with replacement
            N = size(obj.X, 1);
            for i = 1:obj.nTrees
                idx = randsample(N, obj.nSamples, true);
                Xsub = obj.X(idx,:);
                Ysub = obj.Y(idx,:);
                tree = fitctree(Xsub,Ysub, 'PredictorNames', varNames); %train the tree
                obj.trees{i} = tree;
            end
        end

        %Function to make predictions
        function Ypred = predict(obj,Xtest)
            %Use a votation system
            %Distinguish between numeric and cathegoric places. The first
            %is under progress.
            nTest = size(Xtest,1);
            all_preds = cell(nTest, obj.nTrees);

            for i=1:obj.nTrees
                tree = obj.trees{i};
                pred = predict(tree,Xtest);

                if isnumeric(pred)
                    pred = num2cell(pred);
                end
                all_preds(:, i) = pred;
            end

            all_preds_cat = categorical(all_preds);
            Ypred = mode(all_preds_cat,2);
            
            %Under development / I do not know yet if I am going to remove
            %it
            if isnumeric(obj.Y)
                Ypred = double(Ypred);
            end
        end
    end
end

