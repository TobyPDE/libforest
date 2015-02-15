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
    
    CSVDataProvider reader(0);
    reader.read("mnist_train.txt", &storage);
    reader.read("mnist_test.txt", &storageT);
    std::cout << "loaded" << "\n";
    std::cout << storage.getSize() << "\n";
    storage.getClassLabelMap().dump();
    exit(0);
    
    DecisionTreeLearner treeLearner;
    
    treeLearner.autoconf(&storage);
    treeLearner.setUseBootstrap(true);
    
    RandomForestLearner forestLearner;
    
    forestLearner.setTreeLearner(&treeLearner);
    forestLearner.setNumTrees(8);
    forestLearner.setNumThreads(8);    
    
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    RandomForest* forest = forestLearner.learn(&storage);
    //DecisionTree* tree = learner.learn(&storage);
    //RandomForest* forest = new RandomForest;
    /*std::ifstream is("model5.txt");
    forest->read(is);
    is.close();
    */

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

/*    std::ofstream s("modelbin.dat", std::ios::binary);
    if (!s.is_open())
    {
        std::cout << "Eh";
        return 1;
    }
    forest->write(s);
    s.close();
  */  
    std::cout.precision(16);
    std::cout << "\n" << duration/1000000. << "s\n";
    
    //exit(1);
    //RandomForestPrune prune;
    //forest = prune.prune(forest, &storage);
    
    
    {
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
    }

    
    //delete forest;
    return 0;
}
