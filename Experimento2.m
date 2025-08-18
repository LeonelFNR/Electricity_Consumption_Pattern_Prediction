% Now we'll try with a bigger dataset that is in the github of BCAM
% it contains more than 2 classes

%seed
rng(10);

%load dataset
dataset = readtable("dry_bean_dataset.csv");
X = dataset(:,1:end-1); %copy all columns minus the last, because it has the labels
Y = dataset(:,end);

[Y,] = grp2idx(dataset{:,end});

Comparator(X,Y,100);




















