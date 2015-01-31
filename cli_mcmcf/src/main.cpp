#include <iostream>
#include "lib_mcmcf/lib_mcmcf.h"
#include <chrono>
#include <fstream>


using namespace mcmcf;

/**
 * Required tools:
 * - decision trees
 * - random forests
 * - adaboost
 * - gentle boost
 * - mcmc forests
 * - mcmc forests 2
 * - loading CSV
 * - loading SVM
 * - bootstrap
 * - evaluate
 * - save
 */
int main(int c, const char** v)
{
    DataStorage storage;
    DataStorage storageT;
    
    CSVDataProvider reader(0);
    reader.read("mnist_train.txt", &storage);
    reader.read("mnist_test.txt", &storageT);
    std::cout << "loaded" << "\n";
    
    DecisionTreeLearner learner;
    storage.computeIntClassLabels();
    storageT.computeIntClassLabels(&storage);
    learner.autoconf(&storage);
    RandomForestLearner forestLearner;
    learner.setNumBootstrapExamples(10000);
    forestLearner.setTreeLearner(&learner);
    forestLearner.setNumTrees(500);
    forestLearner.setNumThreads(8);    
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    RandomForest* forest = forestLearner.learn(&storage);
    //DecisionTree* tree = learner.learn(&storage);
    
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    std::ofstream s("model.txt");
    forest->write(s);
    s.close();
    
    std::cout.precision(16);
    
    std::cout << "\n" << duration/1000000. << "s\n";
    
    std::vector<int> res;
    forest->classify(&storageT, res);
    
    int error = 0;
    for (int i = 0; i < storageT.getSize(); i++)
    {
        if (res[i] != storageT.getIntClassLabel(i))
        {
            error++;
        }
    }
    
    std::cout << error/static_cast<float>(storageT.getSize()) << "\n";
    
    //delete forest;
    return 0;
}
