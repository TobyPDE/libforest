#include <iostream>
#include "libforest/libforest.h"
#include <chrono>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

using namespace libf;

/**
 * Example of online decision tree learning.
 * 
 * Usage:
 * $ ./examples/cli_online_decision_tree --help
 * Allowed options:
 *   --help                               produce help message
 *   --file-train arg                     path to train DAT file
 *   --file-test arg                      path to test DAT file
 *   --batch-size arg (=1)                number of incoming samples per timestep
 *   --min-split-objective arg (=5)       minimum objective for splitting
 *   --min-split-examples arg (=20)       minimum number of samples for splitting
 *   --min-child-split-examples arg (=10) minimum number of child sampels to split
 *   --num-features arg (=10)             number of features to use (set to 
 *                                        dimensionality of data to learn 
 *                                        deterministically)
 *   --num-thresholds arg (=10)           number of thresholds to use
 *   --max-depth arg (=100)               maximum depth of trees
 */
int main(int argc, const char** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("file-train", boost::program_options::value<std::string>(), "path to train DAT file")
        ("file-test", boost::program_options::value<std::string>(), "path to test DAT file")
        ("batch-size", boost::program_options::value<int>()->default_value(1), "number of incoming samples per timestep")
        ("min-split-objective", boost::program_options::value<float>()->default_value(5.f), "minimum objective for splitting")
        ("min-split-examples", boost::program_options::value<int>()->default_value(20), "minimum number of samples for splitting")
        ("min-child-split-examples", boost::program_options::value<int>()->default_value(10), "minimum number of child sampels to split")
        ("num-features", boost::program_options::value<int>()->default_value(10), "number of features to use (set to dimensionality of data to learn deterministically)")
        ("num-thresholds", boost::program_options::value<int>()->default_value(10), "number of thresholds to use")
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
    
    DataStorage storage;
    DataStorage storageT;
    
    LibforestDataProvider reader;
    reader.read(trainDat.string(), &storageT);
    reader.read(testDat.string(), &storage);
    
    std::cout << "Training Data" << std::endl;
    storageT.dumpInformation();
    
    OnlineDecisionTreeLearner treeLearner;
    
    treeLearner.setThresholdGenerator(RandomThresholdGenerator(storageT));
    treeLearner.setMinSplitObjective(parameters["min-split-objective"].as<float>());
    treeLearner.setMinSplitExamples(parameters["min-split-examples"].as<int>());
    treeLearner.setMinChildSplitExamples(parameters["min-child-split-examples"].as<int>());
    treeLearner.setMaxDepth(parameters["max-depth"].as<int>());
    treeLearner.setNumFeatures(parameters["num-features"].as<int>());
    treeLearner.setNumThresholds(parameters["num-thresholds"].as<int>());
    treeLearner.addCallback(OnlineDecisionTreeLearner::defaultCallback, 1);
    
    const int batchSize = parameters["batch-size"].as<int>();
    
    DecisionTree* tree = 0;
    for (int b = 0; b < storageT.getSize()/batchSize - 1; b++)
    {
        DataStorage batch = storageT.excerpt(b*batchSize, (b + 1)*batchSize - 1);
        tree = treeLearner.learn(&batch, tree);
    }
    AccuracyTool accuracyTool;
    accuracyTool.measureAndPrint(tree, &storageT);
    
    ConfusionMatrixTool confusionMatrixTool;
    confusionMatrixTool.measureAndPrint(tree, &storageT);
    
    delete tree;
    return 0;
}
