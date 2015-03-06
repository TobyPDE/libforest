#include <iostream>
#include "libforest/libforest.h"
#include <chrono>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

using namespace libf;

/**
 * Example of cecision tree learning.
 * 
 * Usage:
 * $ ./lib_forest/examples/cli_decision_tree --help
 * Allowed options:
 *   --help                   produce help message
 *   --file-train arg         path to train DAT file
 *   --file-test arg          path to test DAT file
 *   --num-features arg (=10) number of features to use (set to dimensionality of 
 *                            data to learn deterministically)
 *   --use-bootstrap          use bootstrapping for training
 *   --max-depth arg (=100)   maximum depth of trees
 */
int main(int argc, const char** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("file-train", boost::program_options::value<std::string>(), "path to train DAT file")
        ("file-test", boost::program_options::value<std::string>(), "path to test DAT file")
        ("num-features", boost::program_options::value<int>()->default_value(10), "number of features to use (set to dimensionality of data to learn deterministically)")
        ("use-bootstrap", "use bootstrapping for training")
        ("max-depth", boost::program_options::value<int>()->default_value(100), "maximum depth of trees");

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
    
    boost::filesystem::path trainDat(parameters["file-train"].as<std::string>());
    if (!boost::filesystem::is_regular_file(trainDat))
    {
        std::cout << "Train DAT file does not exist at the specified location." << std::endl;
        return 1;
    }
    
    boost::filesystem::path testDat(parameters["file-test"].as<std::string>());
    if (!boost::filesystem::is_regular_file(testDat))
    {
        std::cout << "Test DAT file does not exist at the specified location." << std::endl;
        return 1;
    }
    
    const bool useBootstrap = parameters.find("use-bootstrap") != parameters.end();
    
    DataStorage storage;
    DataStorage storageT;
    
    LibforestDataProvider reader;
    reader.read(trainDat.string(), &storageT);
    reader.read(testDat.string(), &storage);
    
    std::cout << "Training Data" << std::endl;
    storage.dumpInformation();
    
    DecisionTreeLearner treeLearner;
    
    treeLearner.setUseBootstrap(useBootstrap);
    treeLearner.setMaxDepth(parameters["max-depth"].as<int>());
    treeLearner.setNumFeatures(parameters["num-features"].as<int>());
    treeLearner.addCallback(DecisionTreeLearner::defaultCallback, 1);
    
    DecisionTree* tree = treeLearner.learn(&storage);
    
    AccuracyTool accuracyTool;
    accuracyTool.measureAndPrint(tree, &storageT);
    
    ConfusionMatrixTool confusionMatrixTool;
    confusionMatrixTool.measureAndPrint(tree, &storageT);
    
    delete tree;
    return 0;
}
