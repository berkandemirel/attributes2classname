
clear;
close all;

network = 'CNN_M2K/';
languageModel = 'GloVe/';


numberOfFold = 2;
load('predicateMatrixOnlyTrain.mat');
load([network,'trainScores.mat']);
load('trainImageLabels.mat');
load([languageModel,'trainClassesOnlyTrain.mat']);
load([languageModel,'attributeVectors.mat']);

trainClassList = unique(trainImageLabels);
basePredicateMatrix = predicateMatrix;
baseTrainClasses = trainClasses;

classes = [1:size(trainClassList)];

partitions = cvpartition(classes,'k',numberOfFold);


for i = 1: numberOfFold
    attClassifierResults = [];
    groundTruthLabels = [];
    trainScoresTmp = [];
    trainImageLabelsTmp = [];
    train = partitions.training(i);
    test = partitions.test(i);
    
    predicateMatrix = basePredicateMatrix(train,:);
    trainClasses = baseTrainClasses(train,:);
    testClasses = baseTrainClasses(test,:);
    
    testClassList = trainClassList(test);
    
    for j = 1: size(testClassList)
        
        currClassIndices = find(trainImageLabels == testClassList(j));
        attClassifierResults = [attClassifierResults;trainScores(currClassIndices,:)];
        groundTruthLabels = [groundTruthLabels;j*ones(size(currClassIndices),1)];
        
        
    end
    trainClassListTmp = trainClassList(train);
    
    for j = 1: size(trainClassListTmp)
        
        currClassIndices = find(trainImageLabels == trainClassListTmp(j));
        trainScoresTmp = [trainScoresTmp;trainScores(currClassIndices,:)];
        trainImageLabelsTmp = [trainImageLabelsTmp;j*ones(size(currClassIndices),1)];
        
        
    end
    
    mkdir(['cv_data/',int2str(i)]);
    mkdir(['cv_data/',int2str(i),'/',languageModel]);
    mkdir(['cv_data/',int2str(i),'/',network]);
    save(['cv_data/',int2str(i),'/groundTruthLabels.mat'],'groundTruthLabels');
    save(['cv_data/',int2str(i),'/',network,'/attClassifierResults.mat'],'attClassifierResults');
    save(['cv_data/',int2str(i),'/',languageModel,'trainClasses.mat'],'trainClasses');
    save(['cv_data/',int2str(i),'/',languageModel,'testClasses.mat'],'testClasses');
    save(['cv_data/',int2str(i),'/',languageModel,'attributeVectors.mat'],'attributeVectors');
    save(['cv_data/',int2str(i),'/predicateMatrix.mat'],'predicateMatrix');
    
    trainScores = trainScoresTmp;
    trainImageLabels = trainImageLabelsTmp;
    
    save(['cv_data/',int2str(i),'/',network,'trainScores.mat'],'trainScores');
    save(['cv_data/',int2str(i),'/','trainImageLabels.mat'],'trainImageLabels');
    
    load([network,'trainScores.mat']);
    load('trainImageLabels.mat');
    
    
end

