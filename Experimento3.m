% Now we'll try with a bigger dataset that is in the github of BCAM
% it contains more than 2 classes

%load dataset
opts = detectImportOptions("letter-recognition.csv", "NumHeaderLines",3);
dataset = readtable("letter-recognition.csv",opts); % Adapt to this dataset


X = dataset(:,1:end-1); %copy all columns minus the last, because it has the labels
Y = dataset(:,end);

[Y,] = grp2idx(dataset{:,end});

Comparator(X,Y,100);