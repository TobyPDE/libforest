#include <iostream>
#include "libforest/libforest.h"
#include <chrono>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

using namespace libf;

/**
 * Example of Boosted Random Forest learning on the MNIST datase [1].
 * 
 *  [1] http://yann.lecun.com/exdb/mnist/
 * 
 * **The original MNIST file format is currently not supported.**
 * 
 * Usage:
 * 
 * $ ./examples/cli_adaboost --help
 * Allowed options:
 *   --help                 produce help message
 *   --file-train arg       path to train DAT file
 *   --file-test arg        path to test DAT file
 *   --num-trees arg (=100) number of trees in forest
 *   --max-depth arg (=2)   maximum depth of trees
 */
int main(int argc, const char** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("file-train", boost::program_options::value<std::string>(), "path to train DAT file")
        ("file-test", boost::program_options::value<std::string>(), "path to test DAT file")
        ("num-features", boost::program_options::value<int>()->default_value(10), "number of features to use (set to dimensionality of data to learn deterministically)")
        ("num-trees", boost::program_options::value<int>()->default_value(100), "number of trees in forest")
        ("max-depth", boost::program_options::value<int>()->default_value(2), "maximum depth of trees");
    
    boost::program_options::positional_options_description positionals;
    positionals.add("file-train", 1);
    positionals.add("file-test", 1);
    
    boost::program_options::variables_map parameters;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionals).run(), parameters);
    boost::program_options::notify(parameters);

    if (parameters.find("help") != parameters.end())
    {
        std::cout << desc << std::endl;
        return 1;
    }
    
    boost::filesystem::path mnistTrainDat(parameters["file-train"].as<std::string>());
    if (!boost::filesystem::is_regular_file(mnistTrainDat))
    {
        std::cout << "mnist_train.dat does not exist at the specified location." << std::endl;
        return 1;
    }
    
    boost::filesystem::path mnistTestDat(parameters["file-test"].as<std::string>());
    if (!boost::filesystem::is_regular_file(mnistTestDat))
    {
        std::cout << "mnist_test.dat does not exist at the specified location." << std::endl;
        return 1;
    }
    
    DataStorage storage;
    DataStorage storageT;
    
    LibforestDataProvider reader;
    reader.read(mnistTrainDat.string(), &storageT);
    reader.read(mnistTestDat.string(), &storage);
    
    std::cout << "Training Data" << std::endl;
    storage.dumpInformation();
    
    DecisionTreeLearner treeLearner;
    
    treeLearner.autoconf(&storage);
    treeLearner.setUseBootstrap(false);
    treeLearner.setMaxDepth(parameters["max-depth"].as<int>());
    treeLearner.setNumFeatures(parameters["num-features"].as<int>());
    
    BoostedRandomForestLearner forestLearner;
    forestLearner.addCallback(BoostedRandomForestLearner::defaultCallback, 1);
    
    forestLearner.setTreeLearner(&treeLearner);
    forestLearner.setNumTrees(parameters["num-trees"].as<int>());
    
    BoostedRandomForest* forest = forestLearner.learn(&storage);
    
    AccuracyTool accuracyTool;
    accuracyTool.measureAndPrint(forest, &storageT);
    
    ConfusionMatrixTool confusionMatrixTool;
    confusionMatrixTool.measureAndPrint(forest, &storageT);
    
    delete forest;
    return 0;
}