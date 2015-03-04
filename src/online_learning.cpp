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

void OnlineDecisionTreeLearner::updateSplitStatistics(std::vector<EfficientEntropyHistogram> leftChildStatistics, 
        std::vector<EfficientEntropyHistogram> rightChildStatistics, 
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
    
    // Set up probability distribution for features.
    std::mt19937 g(rd());
    
    OnlineDecisionTreeLearnerState state;
    state.learner = this;
    state.tree = tree;
    state.action = ACTION_UPDATE_TREE;
        
    // Set up a new tree if no existing tree is given.
    if (!tree)
    {
        tree = new DecisionTree(true);
        state.tree = tree;
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
        
        EfficientEntropyHistogram nodeStatistics = tree->getNodeStatistics(leaf);
        std::vector<int> nodeFeatures = tree->getNodeFeatures(leaf);
        std::vector< std::vector<float> > nodeThresholds = tree->getNodeThresholds(leaf);
        std::vector<EfficientEntropyHistogram> leftChildStatistics = tree->getLeftChildStatistics(leaf);
        std::vector<EfficientEntropyHistogram> rightChildStatistics = tree->getRightChildStatistics(leaf);
        
        // This leaf node may be a fresh one.
        if (nodeStatistics.size() == 0)
        {
            nodeStatistics.resize(C);
            
            // Sample thresholds and features.
            std::shuffle(features.begin(), features.end(), std::default_random_engine(rd()));
            for (int f = 0; f < numFeatures; f++)
            {
                nodeFeatures.push_back(features[f]);
            }
            
            // If this is a fresh leaf, then, this is the first sample, so
            // currently there is only one possible threshold.
            nodeThresholds = std::vector< std::vector<float> >(numFeatures);
            for (int f = 0; f < numFeatures; f++)
            {
                nodeThresholds[f].push_back(x_n.first->at(f));
            }
            
            state.action = ACTION_INIT_NODE;
            state.node = leaf;
            state.depth = tree->getDepth(leaf);
            
            evokeCallback(tree, 0, &state);
        }
        else {
            // This is not a fresh node, so add another threshold if not all
            // slots are used.
            // If all slots are used, replace a threshold at random.
            for (int f = 0; f < numFeatures; f++)
            {
                if (nodeThresholds[f].size() < numThresholds)
                {
                    nodeThresholds[f].push_back(x_n.first->at(f));
                }
                else
                {
                    const int t = std::rand() % numThresholds;
                    nodeThresholds[f][t] = x_n.first->at(f);
                }
            }
            
            state.action = ACTION_UPDATE_NODE;
            state.node = leaf;
            state.depth = tree->getDepth(leaf);
            
            evokeCallback(tree, 0, &state);
        }
        
        // Update node statistics.
        nodeStatistics.addOne(x_n.second);
        // Update left and right node statistics for all splits.
        updateSplitStatistics(leftChildStatistics, rightChildStatistics, nodeFeatures, nodeThresholds, x_n);
        
        // As in offline learning, do not split this node
        // - if the number of examples is too small
        // - if the maximum depth is reached
        // TODO: the tree should take care of saving the depth of each node!
        if (nodeStatistics.getMass() < minSplitExamples)
        {
            // Do not split, update leaf histogram according to new sample.
            updateLeafNodeHistogram(tree->getHistogram(leaf), nodeStatistics, smoothingParameter);
            continue;
        }
        
        // Get the best split.
        float bestObjective = 1e35;
        float bestThreshold = -1;
        float bestFeature = -1;
        
        for (int f = 0; f < numFeatures; f++)
        {
            // There may not be numThresholds thresholds yet!
            for (int t = 0; t < nodeThresholds[f].size(); f++)
            {
                const float localObjective = leftChildStatistics[t + numThresholds*f].entropy()
                        + rightChildStatistics[t + numThresholds*f].entropy();
                
                if (localObjective < bestObjective)
                {
                    bestObjective = localObjective;
                    bestThreshold = t;
                    bestFeature = f;
                }
            }
        }
        
        // Split only if the minimum thresholds is obtained.
        if (bestObjective <= minSplitObjective)
        {
            // Do not split, update leaf histogram according to new sample.
            updateLeafNodeHistogram(tree->getHistogram(leaf), nodeStatistics, smoothingParameter);
            continue;
        }
        
        // We split this node!
        tree->setThreshold(leaf, nodeThresholds[bestFeature][bestThreshold]); // Save the actual threshold value.
        tree->setSplitFeature(leaf, nodeFeatures[bestFeature]); // Save the index of the feature.
        tree->splitNode(leaf);
    }
    
    return tree;
}

int OnlineDecisionTreeLearner::defaultCallback(DecisionTree* forest, OnlineDecisionTreeLearnerState* state)
{
    switch (state->action) {
        case OnlineDecisionTreeLearner::ACTION_START_TREE:
            std::cout << "Start decision tree training\n" << "\n";
            break;
        case OnlineDecisionTreeLearner::ACTION_UPDATE_TREE:
            std::cout << "Update decision tree\n" << "\n";
            break;    
        case OnlineDecisionTreeLearner::ACTION_INIT_NODE:
            std::cout << std::setw(15) << std::left << "Init node:"
                    << "depth = " << std::setw(3) << "\n";
            break;
        case OnlineDecisionTreeLearner::ACTION_UPDATE_NODE:
            std::cout << std::setw(15) << std::left << "Update node:"
                    << "depth = " << std::setw(3) << "\n";
            break;
        case OnlineDecisionTreeLearner::ACTION_SPLIT_NODE:
            std::cout << std::setw(15) << std::left << "Split node:"
                    << "depth = " << std::setw(3) << std::right << state->depth
                    << ", objective = " << std::setw(6) << std::left
                    << std::setprecision(4) << state->objective << "\n";
            break;
        default:
            std::cout << "UNKNOWN ACTION CODE " << state->action << "\n";
            break;
    }
    
    return 0;
}