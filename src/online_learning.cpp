#include "libforest/online_learning.h"
#include "libforest/data.h"
#include "libforest/classifiers.h"
#include "libforest/util.h"
#include "fastlog.h"

#include <algorithm>
#include <random>
#include <map>
#include <iomanip>

using namespace libf;

static std::random_device rd;


////////////////////////////////////////////////////////////////////////////////
/// RandomThresholdGenerator
////////////////////////////////////////////////////////////////////////////////

RandomThresholdGenerator::RandomThresholdGenerator(const DataStorage & storage)
{
    const int D = storage.getDimensionality();
    const int N = storage.getSize();
    
    min = std::vector<float>(D, 1e35f);
    max = std::vector<float>(D, -1e35f);
    
    for (int n = 0; n < N; ++n)
    {
        // Retrieve the datapoint to check all features.
        std::pair<DataPoint*, int> x_n = storage[n];
        
        for (int d = 0; d < D; d++)
        {
            if (x_n.first->at(d) < min[d])
            {
                min[d] = x_n.first->at(d);
            }
            if (x_n.first->at(d) > max[d])
            {
                max[d] = x_n.first->at(d);
            }
        }
    }    
}

float RandomThresholdGenerator::sample(int feature)
{
    assert(feature >= 0 && feature < getSize());
    
    std::mt19937 g(rd());
    std::uniform_real_distribution<float> dist(min[feature], max[feature]);
    
    return dist(g);
}

////////////////////////////////////////////////////////////////////////////////
/// OnlineDecisionTreeLearner
////////////////////////////////////////////////////////////////////////////////

inline void updateLeafNodeHistogram(std::vector<float> & leafNodeHistograms, const EfficientEntropyHistogram & hist, float smoothing)
{
    leafNodeHistograms.resize(hist.size());
    
    for (int c = 0; c < hist.size(); c++)
    {
        leafNodeHistograms[c] = std::log((hist.at(c) + smoothing)/(hist.getMass() + hist.size() * smoothing));
    }
}

void OnlineDecisionTreeLearner::updateSplitStatistics(std::vector<EfficientEntropyHistogram> & leftChildStatistics, 
        std::vector<EfficientEntropyHistogram> & rightChildStatistics, 
        const std::vector<int> & features,
        const std::vector< std::vector<float> > & thresholds, 
        const std::pair<DataPoint*, int> & x)
{
    for (int f = 0; f < numFeatures; f++)
    {
        // There may not be numThresholds thresholds yet!!!
        for (int t = 0; t < thresholds[f].size(); t++)
        {
            if (x.first->at(features[f]) < thresholds[f][t])
            {
                // x would fall into left child.
                leftChildStatistics[t + numThresholds*f].addOne(x.second);
            }
            else
            {
                // x would fall into right child.
                rightChildStatistics[t + numThresholds*f].addOne(x.second);
            }
        }
    }
}

DecisionTree* OnlineDecisionTreeLearner::learn(const DataStorage* storage, DecisionTree* tree) {
    
    // Get the number of training examples and the dimensionality of the data set
    const int D = storage->getDimensionality();
    const int C = storage->getClasscount();
    const int N = storage->getSize();
    
    assert(thresholdGenerator.getSize() == D);
    
    // Set up probability distribution for features.
    std::mt19937 g(rd());
    
    // Saves the sum of impurity decrease achieved by each feature
    impurityDecrease = std::vector<float>(D, 0.f);
    
    OnlineDecisionTreeLearnerState state;
    state.action = ACTION_START_TREE;
        
    // Set up a new tree if no existing tree is given.
    if (!tree)
    {
        tree = new DecisionTree(true);
    }
    
    evokeCallback(tree, 0, &state);
    
    // Set up a list of all available features.
    std::vector<int> features(D);
    for (int f = 0; f < D; f++)
    {
        features[f] = f;
    }
    
    for (int n = 0; n < N; n++)
    {
        const std::pair<DataPoint*, int> x_n = (*storage)[n];
        const int leaf = tree->findLeafNode(x_n.first);
        const int depth = tree->getDepth(leaf);
        
        state.node = leaf;
        state.depth = depth;
        
        EfficientEntropyHistogram & nodeStatistics = tree->getNodeStatistics(leaf);
        std::vector<int> & nodeFeatures = tree->getNodeFeatures(leaf);
        std::vector< std::vector<float> > & nodeThresholds = tree->getNodeThresholds(leaf);
        std::vector<EfficientEntropyHistogram> & leftChildStatistics = tree->getLeftChildStatistics(leaf);
        std::vector<EfficientEntropyHistogram> & rightChildStatistics = tree->getRightChildStatistics(leaf);
        
        // This leaf node may be a fresh one.
        if (nodeStatistics.size() <= 0)
        {
            nodeStatistics.resize(C);
            
            leftChildStatistics.resize(numFeatures*numThresholds);
            rightChildStatistics.resize(numFeatures*numThresholds);
            
            nodeFeatures.resize(numFeatures, 0);
            nodeThresholds.resize(numFeatures);
            
            // Sample thresholds and features.
            for (int f = 0; f < numFeatures; f++)
            {
                // As the min and max ranges are generated based on data,
                // we want to filter out irrelevant features.
                do
                {
                    nodeFeatures[f] = features[std::rand()%D];
                }
                while(thresholdGenerator.getMin(nodeFeatures[f]) == thresholdGenerator.getMax(nodeFeatures[f]));
                
                // Check that we did not get a trivial feature.
                assert(thresholdGenerator.getMin(nodeFeatures[f]) != thresholdGenerator.getMax(nodeFeatures[f]));
                        
                nodeThresholds[f].resize(numThresholds);
                
                for (int t = 0; t < numThresholds; t++)
                {
                    nodeThresholds[f][t] = thresholdGenerator.sample(nodeFeatures[f]);
                    
                    if (t > 0)
                    {
                        while (std::abs(nodeThresholds[f][t] - nodeThresholds[f][t - 1]) < 1e-6f)
                        {
                            nodeThresholds[f][t] = thresholdGenerator.sample(nodeFeatures[f]);
                        }
                    }
                    
                    // Initialize left and right child statistic histograms.
                    leftChildStatistics[t + numThresholds*f].resize(C);
                    rightChildStatistics[t + numThresholds*f].resize(C);
                    
                    leftChildStatistics[t + numThresholds*f].reset();
                    rightChildStatistics[t + numThresholds*f].reset();
                    
                    assert(leftChildStatistics[t + numThresholds*f].getMass() == 0);
                    assert(rightChildStatistics[t + numThresholds*f].getMass() == 0);
                }
            }
            
            state.action = ACTION_INIT_NODE;
            evokeCallback(tree, 0, &state);
        }
        
        int K = 1;
        if (useBootstrap)
        {
            std::poisson_distribution<int> poisson(bootstrapLambda);
            int K = std::max(1, poisson(g));
        }
        
        for (int k = 0; k < K; k++)
        {
            // Update node statistics.
            nodeStatistics.addOne(x_n.second);
            // Update left and right node statistics for all splits.
            updateSplitStatistics(leftChildStatistics, rightChildStatistics, nodeFeatures, nodeThresholds, x_n);
        }
            
        // Do not split, update leaf histogram according to new sample.
        updateLeafNodeHistogram(tree->getHistogram(leaf), nodeStatistics, smoothingParameter);
    }
    
    for (int n = 0; n < N; n++)
    {
        const std::pair<DataPoint*, int> x_n = (*storage)[n];
        const int leaf = tree->findLeafNode(x_n.first);
        const int depth = tree->getDepth(leaf);
        
        EfficientEntropyHistogram & nodeStatistics = tree->getNodeStatistics(leaf);
        std::vector<int> & nodeFeatures = tree->getNodeFeatures(leaf);
        std::vector< std::vector<float> > & nodeThresholds = tree->getNodeThresholds(leaf);
        std::vector<EfficientEntropyHistogram> & leftChildStatistics = tree->getLeftChildStatistics(leaf);
        std::vector<EfficientEntropyHistogram> & rightChildStatistics = tree->getRightChildStatistics(leaf);
        
        state.node = leaf;
        state.depth = depth;
        state.samples = nodeStatistics.getMass();
        
        // As in offline learning, do not split this node
        // - if the number of examples is too small
        // - if the maximum depth is reached
        if (nodeStatistics.getMass() < minSplitExamples || depth > maxDepth)
        {
            state.action = ACTION_NOT_SPLITTING_NODE;  
            evokeCallback(tree, 0, &state);
            
            continue;
        }
        
        // Get the best split.
        float bestObjective = 0;
        float bestThreshold = -1;
        float bestFeature = -1;
        
        for (int f = 0; f < numFeatures; f++)
        {
            // There may not be numThresholds thresholds yet!
            for (int t = 0; t < numThresholds; t++)
            {
                // Simple assertion to check consistency:
                // At each timestep, both children together must have
                // the same number of samples as their parent node.
                const int leftMass = leftChildStatistics[t + numThresholds*f].getMass();
                const int rightMass = rightChildStatistics[t + numThresholds*f].getMass();

                assert(leftMass + rightMass == nodeStatistics.getMass());
                
                if (leftMass > minChildSplitExamples || rightMass > minChildSplitExamples)
                {
                    const float localObjective = nodeStatistics.entropy()
                            - leftChildStatistics[t + numThresholds*f].entropy()
                            - rightChildStatistics[t + numThresholds*f].entropy();
                    
                    if (localObjective > bestObjective)
                    {
                        bestObjective = localObjective;
                        bestThreshold = t;
                        bestFeature = f;
                    }
                }
            }
        }
        
        // Split only if the minimum objective is obtained.
        if (bestObjective < minSplitObjective || bestThreshold < 0 || bestFeature < 0)
        {
            state.action = ACTION_NOT_SPLITTING_OBJECTIVE_NODE;  
            state.objective = bestObjective;
            state.minObjective = minSplitObjective;
            
            evokeCallback(tree, 0, &state);
            
            continue;
        }
        
        // Some assertions for the selected split.
        assert(bestFeature < numFeatures
                && bestThreshold < nodeThresholds[bestFeature].size());
        
        // We split this node!
        tree->setThreshold(leaf, nodeThresholds[bestFeature][bestThreshold]); // Save the actual threshold value.
        tree->setSplitFeature(leaf, nodeFeatures[bestFeature]); // Save the index of the feature.
        
        const int leftChild = tree->splitNode(leaf);
        const int rightChild = leftChild  + 1;
        
        // This may be the last sample! So initialize the leaf node histograms!
        updateLeafNodeHistogram(tree->getHistogram(leftChild), 
                leftChildStatistics[bestThreshold + numThresholds*bestFeature], 
                smoothingParameter);
        
        updateLeafNodeHistogram(tree->getHistogram(rightChild), 
                rightChildStatistics[bestThreshold + numThresholds*bestFeature], 
                smoothingParameter);
        
        // Save best objective for variable importance.
        impurityDecrease[bestFeature] += bestObjective;
        
        // Clean up node at this is not a leaf anymore and statistics
        // are not required anymore.
        // nodeStatistics.clear();
        leftChildStatistics.clear();
        rightChildStatistics.clear();
        nodeThresholds.clear();
        nodeFeatures.clear();
        
        // Also clear the histogram as this node is not a leaf anymore!
        tree->getHistogram(leaf).clear();
        
        state.action = ACTION_SPLIT_NODE; 
        state.objective = bestObjective;
        
        evokeCallback(tree, 0, &state);
    }
    
    return tree;
}

int OnlineDecisionTreeLearner::defaultCallback(DecisionTree* tree, OnlineDecisionTreeLearnerState* state)
{
    switch (state->action) {
//        case OnlineDecisionTreeLearner::ACTION_INIT_NODE:
//            std::cout << std::setw(30) << std::left << "Init node: "
//                    << "depth = " << std::setw(6) << state->depth << "\n";
//            break;
        case OnlineDecisionTreeLearner::ACTION_SPLIT_NODE:
            std::cout << std::setw(30) << std::left << "Split node: "
                    << "depth = " << std::setw(6) << state->depth
                    << "samples = " << std::setw(6) << state->samples
                    << "objective = " << std::setw(6)
                    << std::setprecision(3) << state->objective << "\n";
            break;
    }
    
    return 0;
}

int OnlineDecisionTreeLearner::verboseCallback(DecisionTree* tree, OnlineDecisionTreeLearnerState* state)
{
    switch (state->action) {
        case OnlineDecisionTreeLearner::ACTION_START_TREE:
            std::cout << "Start decision tree training." << "\n";
            break; 
        case OnlineDecisionTreeLearner::ACTION_INIT_NODE:
            std::cout << std::setw(30) << std::left << "Init node: "
                    << "depth = " << std::setw(6) << state->depth << "\n";
            break;
        case OnlineDecisionTreeLearner::ACTION_NOT_SPLITTING_NODE:
            std::cout << std::setw(30) << std::left << "Not splitting node: "
                    << "depth = " << std::setw(6) << state->depth
                    << "samples = " << std::setw(6) << state->samples << "\n";
            break;
        case OnlineDecisionTreeLearner::ACTION_NOT_SPLITTING_OBJECTIVE_NODE:
            std::cout << std::setw(30) << std::left << "Not splitting node: "
                    << "depth = " << std::setw(6) << state->depth
                    << "samples = " << std::setw(6) << state->samples
                    << "objective = " << std::setw(6)
                    << std::setprecision(3) << state->objective
                    << "min objective = " << std::setw(6)
                    << std::setprecision(3) << state->minObjective << "\n";
            break;
        case OnlineDecisionTreeLearner::ACTION_SPLIT_NODE:
            std::cout << std::setw(30) << std::left << "Split node: "
                    << "depth = " << std::setw(6) << state->depth
                    << "samples = " << std::setw(6) << state->samples
                    << "objective = " << std::setw(6)
                    << std::setprecision(3) << state->objective << "\n";
            break;
        default:
            std::cout << "UNKNOWN ACTION CODE " << state->action << "\n";
            break;
    }
    
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// RandomForestLearner
////////////////////////////////////////////////////////////////////////////////

RandomForest* OnlineRandomForestLearner::learn(const DataStorage* storage, RandomForest* forest)
{
    const int D = storage->getDimensionality();
    
    if (!forest)
    {
        forest = new RandomForest();
    }
    
    // Initialize variable importance values.
    impurityDecrease = std::vector<float>(D, 0.f);
    
    // Set up the state for the call backs
    OnlineRandomForestLearnerState state;
    state.tree = 0;
    state.numTrees = this->getNumTrees();
    state.action = ACTION_START_FOREST;
    
    evokeCallback(forest, 0, &state);
    
    int treeStartCounter = 0; 
    int treeFinishCounter = 0; 
    
    DecisionTree* tree;
    #pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < numTrees; i++)
    {
        #pragma omp critical
        {
            if (i >= forest->getSize())
            {
                tree = new DecisionTree(true);
                forest->addTree(tree);
            }
            else {
                tree = forest->getTree(i);
            }
            
            state.tree = ++treeStartCounter;
            state.action = ACTION_START_TREE;
            evokeCallback(forest, treeStartCounter - 1, &state);
        }
        
        treeLearner->learn(storage, tree);
        
        #pragma omp critical
        {
            state.tree = ++treeFinishCounter;
            state.action = ACTION_FINISH_TREE;
            evokeCallback(forest, treeFinishCounter - 1, &state);

            // Update variable importance.
            for (int f = 0; f < D; ++f)
            {
                impurityDecrease[f] += treeLearner->getMDIImportance(f)/this->numTrees;
            }
        }
    }
    
    state.tree = 0;
    state.action = ACTION_FINISH_FOREST;
    evokeCallback(forest, 0, &state);
    
    return forest;
}

int OnlineRandomForestLearner::defaultCallback(RandomForest* forest, OnlineRandomForestLearnerState* state)
{
    switch (state->action) {
//        case RandomForestLearner::ACTION_START_FOREST:
//            std::cout << "Start random forest training" << "\n";
//            break;
        case RandomForestLearner::ACTION_FINISH_FOREST:
            std::cout << "Finished forest in " << state->getPassedTime().count()/1000000. << "s\n";
            break;
    }
    
    return 0;
}

int OnlineRandomForestLearner::verboseCallback(RandomForest* forest, OnlineRandomForestLearnerState* state)
{
    switch (state->action) {
        case RandomForestLearner::ACTION_START_FOREST:
            std::cout << "Start random forest training" << "\n";
            break;
        case RandomForestLearner::ACTION_START_TREE:
            std::cout << std::setw(15) << std::left << "Start tree " 
                    << std::setw(4) << std::right << state->tree 
                    << " out of " 
                    << std::setw(4) << state->numTrees << "\n";
            break;
        case RandomForestLearner::ACTION_FINISH_TREE:
            std::cout << std::setw(15) << std::left << "Finish tree " 
                    << std::setw(4) << std::right << state->tree 
                    << " out of " 
                    << std::setw(4) << state->numTrees << "\n";
            break;
        case RandomForestLearner::ACTION_FINISH_FOREST:
            std::cout << "Finished forest in " << state->getPassedTime().count()/1000000. << "s\n";
            break;
        default:
            std::cout << "UNKNOWN ACTION CODE " << state->action << "\n";
            break;
    }
    
    return 0;
}