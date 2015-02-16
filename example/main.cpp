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
    reader.read("mnist_train.dat", &storage);
    reader.read("mnist_test.dat", &storageT);
    
    DecisionTreeLearner treeLearner;
    
    treeLearner.autoconf(&storage);
    treeLearner.setUseBootstrap(true);
    treeLearner.setMinSplitExamples(5);
    
    RandomForestLearner forestLearner;
    
    forestLearner.addCallback(RandomForestLearner::defaultCallback, 1);
    
    forestLearner.setTreeLearner(&treeLearner);
    forestLearner.setNumTrees(1);
    forestLearner.setNumThreads(8);    
    
    RandomForestLearnerState state;
    RandomForest* forest = forestLearner.learn(&storage);
    std::cout << state.getPassedTime().count()/1000000.0f << "s\n";
    
    std::vector<int> res;
    forest->classify(&storageT, res);

    int error = 0;
    for (int i = 0; i < storageT.getSize(); i++)
    {
        if (res[i] != storageT.getClassLabel(i))
        {
            error++;
        }
    }

    std::cout << error/static_cast<float>(storageT.getSize()) << "\n";
    
    delete forest;
    return 0;
}
