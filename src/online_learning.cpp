#include "libforest/online_learning.h"
#include "libforest/data.h"
#include "libforest/classifiers.h"
#include "libforest/util.h"

#include <algorithm>
#include <random>
#include <map>
#include <iomanip>

using namespace libf;

static std::random_device rd;
static std::mt19937 g(rd());

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
        DataPoint* x = storage.getDataPoint(n);
        
        for (int d = 0; d < D; d++)
        {
            if (x->at(d) < min[d])
            {
                min[d] = x->at(d);
            }
            if (x->at(d) > max[d])
            {
                max[d] = x->at(d);
            }
        }
    }    
}

float RandomThresholdGenerator::sample(int feature)
{
    // assert(feature >= 0 && feature < getSize());
    std::uniform_real_distribution<float> dist(min[feature], max[feature]);
    
    return dist(g);
}

////////////////////////////////////////////////////////////////////////////////
/// OnlineDecisionTreeLearner
////////////////////////////////////////////////////////////////////////////////

inline void updateLeafNodeHistogram(std::vector<float> & leafNodeHistograms, const EfficientEntropyHistogram & hist, float smoothing)
{
    leafNodeHistograms.resize(hist.getSize());
    
    for (int c = 0; c < hist.getSize(); c++)
    {
        leafNodeHistograms[c] = std::log((hist.at(c) + smoothing)/(hist.getMass() + hist.getSize() * smoothing));
    }
}

void OnlineDecisionTreeLearner::updateSplitStatistics(std::vector<EfficientEntropyHistogram> & leftChildStatistics, 
        std::vector<EfficientEntropyHistogram> & rightChildStatistics, 
        const std::vector<int> & features,
        const std::vector< std::vector<float> > & thresholds, 
        const DataPoint* x, const int label)
{
    for (int f = 0; f < numFeatures; f++)
    {
        // There may not be numThresholds thresholds yet!!!
        for (unsigned int t = 0; t < thresholds[f].size(); t++)
        {
            if (x->at(features[f]) < thresholds[f][t])
            {
                // x would fall into left child.
                leftChildStatistics[t + numThresholds*f].addOne(label);
            }
            else
            {
                // x would fall into right child.
                rightChildStatistics[t + numThresholds*f].addOne(label);
            }
        }
    }
}

DecisionTree* OnlineDecisionTreeLearner::learn(const DataStorage* storage)
{
    DecisionTree * tree = new DecisionTree();
    tree->addNode(0);
    
    return learn(storage, tree);
}

DecisionTree* OnlineDecisionTreeLearner::learn(const DataStorage* storage, DecisionTree* tree)
{
    
    // Get the number of training examples and the dimensionality of the data set
    const int D = storage->getDimensionality();
    const int C = storage->getClasscount();
    const int N = storage->getSize();
    
    assert(numFeatures <= D);
    assert(thresholdGenerator.getSize() == D);
    
    // The tree must have at least the root note!
    assert(tree->getNumNodes() > 0);
    
    // Saves the sum of impurity decrease achieved by each feature
    importance = std::vector<float>(D, 0.f);
    
    OnlineDecisionTreeLearnerState state;
    state.action = ACTION_START_TREE;
    
    evokeCallback(tree, 0, &state);
    
    // Set up a list of all available features.
    numFeatures = std::min(numFeatures, D);
    std::vector<int> features(D);
    for (int f = 0; f < D; f++)
    {
        features[f] = f;
    }
    
    for (int n = 0; n < N; n++)
    {
        const DataPoint* x = storage->getDataPoint(n);
        const int label = storage->getClassLabel(n);
        const int leaf = tree->findLeafNode(x);
        const int depth = tree->getDepth(leaf);
        
        state.node = leaf;
        state.depth = depth;
        
        EfficientEntropyHistogram & nodeStatistics = tree->getNodeStatistics(leaf);
        std::vector<int> & nodeFeatures = tree->getNodeFeatures(leaf);
        std::vector< std::vector<float> > & nodeThresholds = tree->getNodeThresholds(leaf);
        std::vector<EfficientEntropyHistogram> & leftChildStatistics = tree->getLeftChildStatistics(leaf);
        std::vector<EfficientEntropyHistogram> & rightChildStatistics = tree->getRightChildStatistics(leaf);
        
        // This leaf node may be a fresh one.
        if (nodeStatistics.getSize() <= 0)
        {
            nodeStatistics.resize(C);
            
            leftChildStatistics.resize(numFeatures*numThresholds);
            rightChildStatistics.resize(numFeatures*numThresholds);
            
            nodeFeatures.resize(numFeatures, 0);
            nodeThresholds.resize(numFeatures);
            
            // Sample thresholds and features.
            std::shuffle(features.begin(), features.end(), std::default_random_engine(rd()));
            
            // Used to make sure, that non/trivial, different features are chosen.
//            int f_alt = numFeatures;
            
            for (int f = 0; f < numFeatures; f++)
            {
                // Try the first feature.
                nodeFeatures[f] = features[f];
                assert(nodeFeatures[f] >= 0 && nodeFeatures[f] < D);
                
                // This may be a trivial feature, so search for the next non/trivial
                // feature; make sure that all chosen features are different.
                const int M = 10;
                int m = 0;

                // TODO: this should not be necessary!
//                while(thresholdGenerator.getMin(nodeFeatures[f]) == thresholdGenerator.getMax(nodeFeatures[f])
//                        && m < M && f_alt < D)
//                {
//                    nodeFeatures[f] = features[f_alt];
//                    ++f_alt;
//                    ++m;
//                }
                        
                nodeThresholds[f].resize(numThresholds);
                
                for (int t = 0; t < numThresholds; t++)
                {
                    nodeThresholds[f][t] = thresholdGenerator.sample(nodeFeatures[f]);
                    
                    if (t > 0)
                    {
                        // Maximum 10 tries to get a better threshold.
                        m = 0;
                        while (std::abs(nodeThresholds[f][t] - nodeThresholds[f][t - 1]) < 1e-6f
                                && m < M)
                        {
                            nodeThresholds[f][t] = thresholdGenerator.sample(nodeFeatures[f]);
                            ++m;
                        }
                    }
                    
                    // Initialize left and right child statistic histograms.
                    leftChildStatistics[t + numThresholds*f].resize(C);
                    rightChildStatistics[t + numThresholds*f].resize(C);
                    
                    leftChildStatistics[t + numThresholds*f].reset();
                    rightChildStatistics[t + numThresholds*f].reset();
                }
            }
            
            state.action = ACTION_INIT_NODE;
            evokeCallback(tree, 0, &state);
        }
        
        int K = 1;
        if (useBootstrap)
        {
            std::poisson_distribution<int> poisson(bootstrapLambda);
            K = poisson(g); // May also give zero.
        }
        
        for (int k = 0; k < K; k++)
        {
            // Update node statistics.
            nodeStatistics.addOne(label);
            // Update left and right node statistics for all splits.
            updateSplitStatistics(leftChildStatistics, rightChildStatistics, 
                    nodeFeatures, nodeThresholds, x, label);
        }
        
        state.node = leaf;
        state.depth = depth;
        state.samples = nodeStatistics.getMass();
        
        // As in offline learning, do not split this node
        // - if the number of examples is too small
        // - if the maximum depth is reached
        if (nodeStatistics.getMass() < minSplitExamples || nodeStatistics.isPure() 
                || depth >= maxDepth)
        {
            // Do not split, update leaf histogram according to new sample.
            updateLeafNodeHistogram(tree->getHistogram(leaf), nodeStatistics, smoothingParameter);
        
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
            for (int t = 0; t < numThresholds; t++)
            {
                const int leftMass = leftChildStatistics[t + numThresholds*f].getMass();
                const int rightMass = rightChildStatistics[t + numThresholds*f].getMass();
                
                if (leftMass > minChildSplitExamples && rightMass > minChildSplitExamples)
                {
                    const float localObjective = nodeStatistics.getEntropy()
                            - leftChildStatistics[t + numThresholds*f].getEntropy()
                            - rightChildStatistics[t + numThresholds*f].getEntropy();
                    
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
        if (bestObjective < minSplitObjective)
        {
            // Do not split, update leaf histogram according to new sample.
            updateLeafNodeHistogram(tree->getHistogram(leaf), 
                    nodeStatistics, smoothingParameter);
        
            state.action = ACTION_NOT_SPLITTING_OBJECTIVE_NODE;  
            state.objective = bestObjective;
            state.minObjective = minSplitObjective;
            
            evokeCallback(tree, 0, &state);
            
            continue;
        }
        
        assert(bestFeature >= 0 && nodeFeatures[bestFeature] >= 0 
                && nodeFeatures[bestFeature] < D);
        
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
        ++importance[bestFeature];
        
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

RandomForest* OnlineRandomForestLearner::learn(const DataStorage* storage)
{
    RandomForest * forest = new RandomForest();
    return learn(storage, forest);
}

RandomForest* OnlineRandomForestLearner::learn(const DataStorage* storage, RandomForest* forest)
{
    const int D = storage->getDimensionality();
    
    for (int i = 0; i < numTrees; i++)
    {
        if (i >= forest->getSize())
        {
            DecisionTree* tree = new DecisionTree(true);
            tree->addNode(0);
            
            forest->addTree(tree);
        }
    }
    
    // Initialize variable importance values.
    importance = std::vector<float>(D, 0.f);
    
    // Set up the state for the call backs
    OnlineRandomForestLearnerState state;
    state.tree = 0;
    state.numTrees = this->getNumTrees();
    state.action = ACTION_START_FOREST;
    
    evokeCallback(forest, 0, &state);
    
    int treeStartCounter = 0; 
    int treeFinishCounter = 0; 
    
    #pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < numTrees; i++)
    {
        #pragma omp critical
        {
            state.tree = ++treeStartCounter;
            state.action = ACTION_START_TREE;
            
            evokeCallback(forest, treeStartCounter - 1, &state);
        }
        
        DecisionTree* tree = forest->getTree(i);
        treeLearner->learn(storage, tree);
        
        #pragma omp critical
        {
            state.tree = ++treeFinishCounter;
            state.action = ACTION_FINISH_TREE;
            
            evokeCallback(forest, treeFinishCounter - 1, &state);

            // Update variable importance.
            for (int f = 0; f < D; ++f)
            {
                importance[f] += treeLearner->getImportance(f);
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
        case RandomForestLearner::ACTION_START_FOREST:
            std::cout << "Start random forest training" << "\n";
            break;
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