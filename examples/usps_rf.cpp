#include <iostream>
#include "libforest/libforest.h"
#include <chrono>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

using namespace libf;

/**
 * Example of Random Forest learning on the USPS datase [1].
 * 
 *  [1] http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html
 * 
 * Download both train and test set and extract; the files are in
 * whitespace separated format readable using CSVDataProvider.
 * 
 * Usage:
 * 
 * $ ./lib_forest/example/cli_mnist_rf --help
 * Allowed options:
 *   --help                 produce help message
 *   --usps-train arg       path to usps train CSV file
 *   --usps-test arg        path to usps test CSV file
 *   --num-trees arg (=100) number of trees in forest
 *   --max-depth arg (=100) maximum depth of trees
 *   --num-threads arg (=1) number of threads for learning
 */
int main(int argc, const char** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("usps-train", boost::program_options::value<std::string>(), "path to usps train CSV file")
        ("usps-test", boost::program_options::value<std::string>(), "path to usps test CSV file")
        ("num-features", boost::program_options::value<int>()->default_value(10), "number of features to use (set to dimensionality of data to learn deterministically)")
        ("num-trees", boost::program_options::value<int>()->default_value(100), "number of trees in forest")
        ("max-depth", boost::program_options::value<int>()->default_value(100), "maximum depth of trees")
        ("num-threads", boost::program_options::value<int>()->default_value(1), "number of threads for learning");

    boost::program_options::positional_options_description positionals;
    positionals.add("usps-train", 1);
    positionals.add("usps-test", 1);
    
    boost::program_options::variables_map parameters;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionals).run(), parameters);
    boost::program_options::notify(parameters);

    if (parameters.find("help") != parameters.end())
    {
        std::cout << desc << std::endl;
        return 1;
    }
    
    boost::filesystem::path uspsTrainDat(parameters["usps-train"].as<std::string>());
    if (!boost::filesystem::is_regular_file(uspsTrainDat))
    {
        std::cout << "USPS train file does not exist at the specified location." << std::endl;
        return 1;
    }
    
    boost::filesystem::path uspsTestDat(parameters["usps-test"].as<std::string>());
    if (!boost::filesystem::is_regular_file(uspsTestDat))
    {
        std::cout << "USPS test file does not exist at the specified location." << std::endl;
        return 1;
    }
    
    DataStorage storage;
    DataStorage storageT;
    
    CSVDataProvider reader(0, " ");
    reader.read(uspsTrainDat.string(), &storageT);
    reader.read(uspsTestDat.string(), &storage);
    
    std::cout << "Training Data" << std::endl;
    storage.dumpInformation();
    
    DecisionTreeLearner treeLearner;
    
    treeLearner.autoconf(&storage);
    treeLearner.setUseBootstrap(false);
    treeLearner.setMaxDepth(parameters["max-depth"].as<int>());
    treeLearner.setNumFeatures(parameters["num-features"].as<int>());
    
    RandomForestLearner forestLearner;
    forestLearner.addCallback(RandomForestLearner::defaultCallback, 1);
    
    forestLearner.setTreeLearner(&treeLearner);
    forestLearner.setNumTrees(parameters["num-trees"].as<int>());
    forestLearner.setNumThreads(parameters["num-threads"].as<int>());
    
    RandomForest* forest = forestLearner.learn(&storage);
    
    AccuracyTool accuracyTool;
    accuracyTool.measureAndPrint(forest, &storageT);
    
    ConfusionMatrixTool confusionMatrixTool;
    confusionMatrixTool.measureAndPrint(forest, &storageT);
    
    delete forest;
    return 0;
}
