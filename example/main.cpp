#include <iostream>
#include "libforest/libforest.h"
#include <chrono>
#include <fstream>


using namespace libf;

/**
 * Required tools:
 * - decision trees
 * - Ferns
 * - random forests
 * - adaboost
 * - loading CSV
 * - loading SVM
 * - bootstrap
 * - evaluate
 * - save
 * 
Loading:
CSV
LIBSVM
MATLAB (Only if installed)
OpenCV (Only if installed)
Libforest Binary

Saving:
CSV
LIBSVM
Libforest Binary
OpenCV (Only if installed)

Classifiers:
Decision Tree -> Add histogram to leaf nodes
 * Random Forest -> Add Bayes Smoothing
Weighted Ensemble -> Add 

Learning Algorithms:
Tree Learning
Fern Learning
Random Forest Learning
Multiclass AdaBoost
Forest Pruning (Classification error, Modified Hamming Distanceaa)

Data Types:
Class labels: string
Feature Vectors:
- float
- categorical
- pairs

Evaluation Methods:
Classification Error
Multiclass F Score
In-Class Errors
Confusion Matrix
 * 
 * NodeJS shortcut function:
 * evalPerf({
 *  trainingSet : {
 *  file: "test.csv", 
 * type: "CSV"
 * }
 * testSet : {
 *  file: "test.csv", 
 * type: "CSV"
 * }, 
 * classifier: "RandomForest" ("DecisionTree", "Fern", "AdaBoost", "PrunedRandomForest")
 * })
 * 
 * storage = libf.DataStorage.create(data, labels);
 * forest = libf.RandomForest.learn(storage, {
 *  algorithm: "default", "AdaBoost", 
 *  prune: true, 
 * })
 * tree = libf.DecisionTree.learn(storage, {
 * })
 * fern = libf.Fern.learn(storage, {
 * })
 */

int main(int c, const char** v)
{
    DataStorage storage;
    DataStorage storageT;
    
    LibforestDataProvider reader;
    reader.read("/Users/Toby/Projects/libforest/example/build/mnist_train.dat", &storageT);
    reader.read("/Users/Toby/Projects/libforest/example/build/mnist_test.dat", &storage);
    
    DecisionTreeLearner treeLearner;
    
    treeLearner.autoconf(&storage);
    treeLearner.setUseBootstrap(false);
    treeLearner.setMaxDepth(1);
    
    BoostedRandomForestLearner forestLearner;
    forestLearner.addCallback(BoostedRandomForestLearner::defaultCallback, 1);
    
    forestLearner.setTreeLearner(&treeLearner);
    forestLearner.setNumTrees(100);
    
    BoostedRandomForest* forest = forestLearner.learn(&storage);
    
    AccuracyTool accuracyTool;
    accuracyTool.measureAndPrint(forest, &storageT);
    
    ConfusionMatrixTool confusionMatrixTool;
    confusionMatrixTool.measureAndPrint(forest, &storageT);
    
    delete forest;
    return 0;
}
