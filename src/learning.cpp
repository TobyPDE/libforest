#include "libforest/learning.h"
#include "libforest/data.h"
#include "libforest/classifiers.h"
#include "libforest/util.h"

#include <algorithm>
#include <random>
#include <map>
#include <iomanip>
#include <queue>
#include <stack>

using namespace libf;

static std::random_device rd;

////////////////////////////////////////////////////////////////////////////////
/// DecisionTreeLearner
////////////////////////////////////////////////////////////////////////////////

/**
 * Updates the leaf node histograms using a smoothing parameter
 */
inline void updateLeafNodeHistogram(std::vector<float> & leafNodeHistogram, const EfficientEntropyHistogram & hist, float smoothing, bool useBootstrap)
{
    const int C = hist.getSize();
    
    leafNodeHistogram.resize(C);
    BOOST_ASSERT(leafNodeHistogram.size() > 0);
    
    if(!useBootstrap)
    {
        for (int c = 0; c < C; c++)
        {
            leafNodeHistogram[c] = std::log((hist.at(c) + smoothing)/(hist.getMass() + hist.getSize() * smoothing));
        }
    }
}

DecisionTree::ptr DecisionTreeLearner::learn(AbstractDataStorage::ptr dataStorage)
{
    BOOST_ASSERT(numFeatures <= dataStorage->getDimensionality());
    
    AbstractDataStorage::ptr storage;
    // If we use bootstrap sampling, then this array contains the results of 
    // the sampler. We use it later in order to refine the leaf node histograms
    std::vector<bool> sampled;
    
    if (useBootstrap)
    {
        storage = dataStorage->bootstrap(numBootstrapExamples, sampled);
    }
    else
    {
        storage = dataStorage;
    }
    
    // Get the number of training examples and the dimensionality of the data set
    const int D = storage->getDimensionality();
    const int C = storage->getClasscount();
    
    // Set up a new tree. 
    DecisionTree::ptr tree = std::make_shared<DecisionTree>();
    tree->addNode();
    
    // Set up the state for the call backs
    DecisionTreeLearnerState state;
    state.action = ACTION_START_TREE;
    
    evokeCallback(tree, 0, state);
    
    // This is the list of nodes that still have to be split
    std::vector<int> splitStack;
    splitStack.reserve(static_cast<int>(fastlog2(storage->getSize())));
    
    // Add the root node to the list of nodes that still have to be split
    splitStack.push_back(0);
    
    // This matrix stores the training examples for certain nodes. 
    std::vector< int* > trainingExamples;
    std::vector< int > trainingExamplesSizes;
    trainingExamples.reserve(LIBF_GRAPH_BUFFER_SIZE);
    trainingExamplesSizes.reserve(LIBF_GRAPH_BUFFER_SIZE);
    
    // Saves the sum of impurity decrease achieved by each feature
    importance = std::vector<float>(D, 0.f);
    
    // Add all training example to the root node
    trainingExamplesSizes.push_back(storage->getSize());
    trainingExamples.push_back(new int[trainingExamplesSizes[0]]);
    for (int n = 0; n < storage->getSize(); n++)
    {
        trainingExamples[0][n] = n;
    }
    
    // We use these arrays during training for the left and right histograms
    EfficientEntropyHistogram leftHistogram(C);
    EfficientEntropyHistogram rightHistogram(C);
    
    // We use this in order to sort the data points
    FeatureComparator cp;
    cp.storage = storage;
    
    // Set up a probability distribution over the features
    std::mt19937 g(rd());
    // Set up the array of possible features, we use it in order to sample
    // the features without replacement
    std::vector<int> sampledFeatures(D);
    for (int d = 0; d < D; d++)
    {
        sampledFeatures[d] = d;
    }

    // Start training
    while (splitStack.size() > 0)
    {
        // Extract an element from the queue
        const int node = splitStack.back();
        splitStack.pop_back();
        
        // Get the training example list
        int* trainingExampleList = trainingExamples[node];
        const int N = trainingExamplesSizes[node];

        // Set up the right histogram
        // Because we start with the threshold being at the left most position
        // The right child node contains all training examples
        
        EfficientEntropyHistogram hist(C);
        for (int m = 0; m < N; m++)
        {
            // Get the class label of this training example
            hist.addOne(storage->getClassLabel(trainingExampleList[m]));
        }

        // Don't split this node
        //  If the number of examples is too small
        //  If the training examples are all of the same class
        //  If the maximum depth is reached
        if (hist.getMass() < minSplitExamples || hist.isPure() || tree->getNodeConfig(node).getDepth() >= maxDepth)
        {
            // Resize and initialize the leaf node histogram
            updateLeafNodeHistogram(tree->getNodeData(node).histogram, hist, smoothingParameter, useBootstrap);
            BOOST_ASSERT(tree->getNodeData(node).histogram.size() > 0);
            continue;
        }
        
        // These are the parameters we optimize
        float bestThreshold = 0;
        int bestFeature = -1;
        float bestObjective = 1e35;
        int bestLeftMass = 0;
        int bestRightMass = N;

        // Sample random features
        std::shuffle(sampledFeatures.begin(), sampledFeatures.end(), std::default_random_engine(rd()));
        
        // Optimize over all features
        for (int f = 0; f < numFeatures; f++)
        {
            const int feature = sampledFeatures[f];
            
            cp.feature = feature;
            std::sort(trainingExampleList, trainingExampleList + N, cp);
            
            // Initialize the histograms
            leftHistogram.reset();
            rightHistogram = hist;
            
            float leftValue = storage->getDataPoint(trainingExampleList[0])(feature);
            int leftClass = storage->getClassLabel(trainingExampleList[0]);
            
            // Test different thresholds
            // Go over all examples in this node
            for (int m = 1; m < N; m++)
            {
                const int n = trainingExampleList[m];
                
                // Move the last point to the left histogram
                leftHistogram.addOne(leftClass);
                rightHistogram.subOne(leftClass);
                        
                // It does
                // Get the two feature values
                const float rightValue = storage->getDataPoint(n)(feature);
                
                // Skip this split, if the two points lie too close together
                const float diff = rightValue - leftValue;
                
                if (diff < 1e-6f)
                {
                    leftValue = rightValue;
                    leftClass = storage->getClassLabel(n);
                    continue;
                }
                
                // Get the objective function
                const float localObjective = leftHistogram.getEntropy()
                        + rightHistogram.getEntropy();
                
                if (localObjective < bestObjective)
                {
                    // Get the threshold value
                    bestThreshold = (leftValue + rightValue);
                    bestFeature = feature;
                    bestObjective = localObjective;
                    bestLeftMass = leftHistogram.getMass();
                    bestRightMass = rightHistogram.getMass();
                }
                
                leftValue = rightValue;
                leftClass = storage->getClassLabel(n);
            }
        }
        
        // We spare the additional multiplication at each iteration.
        bestThreshold *= 0.5f;
        
        // Did we find good split values?
        if (bestFeature < 0 || bestLeftMass < minChildSplitExamples || bestRightMass < minChildSplitExamples)
        {
            // We didn't
            // Don't split
            updateLeafNodeHistogram(tree->getNodeData(node).histogram, hist, smoothingParameter, useBootstrap);
            BOOST_ASSERT(tree->getNodeData(node).histogram.size() > 0);
            continue;
        }
        
        // Set up the data lists for the child nodes
        trainingExamplesSizes.push_back(bestLeftMass);
        trainingExamplesSizes.push_back(bestRightMass);
        trainingExamples.push_back(new int[bestLeftMass]);
        trainingExamples.push_back(new int[bestRightMass]);
        
        int* leftList = trainingExamples[trainingExamples.size() - 2];
        int* rightList = trainingExamples[trainingExamples.size() - 1];
        
        // Sort the points
        for (int m = 0; m < N; m++)
        {
            const int n = trainingExampleList[m];
            const float featureValue = storage->getDataPoint(n)(bestFeature);
            
            if (featureValue < bestThreshold)
            {
                leftList[--bestLeftMass] = n;
            }
            else
            {
                rightList[--bestRightMass] = n;
            }
        }
        
        // Ok, split the node
        tree->getNodeConfig(node).setThreshold(bestThreshold);
        tree->getNodeConfig(node).setSplitFeature(bestFeature);
        const int leftChild = tree->splitNode(node);
        
        state.action = ACTION_SPLIT_NODE;
        state.depth = tree->getNodeConfig(node).getDepth();
        state.objective = bestObjective;
        
        evokeCallback(tree, 0, state);
        
        // Save the impurity reduction for this feature if requested
        importance[bestFeature] += N/storage->getSize()*(hist.getEntropy() - bestObjective);
        
        // Prepare to split the child nodes
        splitStack.push_back(leftChild);
        splitStack.push_back(leftChild + 1);
        
        delete[] trainingExampleList;
    }
    
    // If we use bootstrap, we use all the training examples for the 
    // histograms
    if (useBootstrap)
    {
        updateHistograms(tree, dataStorage);
    }
    
    return tree;
}

void DecisionTreeLearner::updateHistograms(DecisionTree::ptr tree, AbstractDataStorage::ptr storage) const
{
    const int C = storage->getClasscount();
    
    // Reset all histograms
    for (int v = 0; v < tree->getNumNodes(); v++)
    {
        if (tree->getNodeConfig(v).isLeafNode())
        {
            std::vector<float> & hist = tree->getNodeData(v).histogram;
            
            // Make sure that hist is initialized.
            hist.resize(C);
            
            for (int c = 0; c < C; c++)
            {
                hist[c] = 0;
            }
        }
    }
    
    
    // Compute the weights for each data point
    for (int n = 0; n < storage->getSize(); n++)
    {
        int leafNode = tree->findLeafNode(storage->getDataPoint(n));
        tree->getNodeData(leafNode).histogram[storage->getClassLabel(n)] += 1;
    }
    
    // Normalize the histograms
    for (int v = 0; v < tree->getNumNodes(); v++)
    {
        if (tree->getNodeConfig(v).isLeafNode())
        {
            std::vector<float> & hist = tree->getNodeData(v).histogram;
            float total = 0;
            for (int c = 0; c < C; c++)
            {
                total += hist[c];
            }
            for (int c = 0; c < C; c++)
            {
                hist[c] = std::log((hist[c] + smoothingParameter)/(total + C*smoothingParameter));
            }
        }
    }
}

int DecisionTreeLearner::defaultCallback(DecisionTree::ptr tree, const DecisionTreeLearnerState & state)
{
    switch (state.action) {
        case DecisionTreeLearner::ACTION_START_TREE:
            std::cout << "Start decision tree training" << "\n";
            break;
        case DecisionTreeLearner::ACTION_SPLIT_NODE:
            std::cout << std::setw(15) << std::left << "Split node:"
                    << "depth = " << std::setw(3) << std::right << state.depth
                    << ", objective = " << std::setw(6) << std::left
                    << std::setprecision(4) << state.objective << "\n";
            break;
        default:
            std::cout << "UNKNOWN ACTION CODE " << state.action << "\n";
            break;
    }
    
    return 0;

}


////////////////////////////////////////////////////////////////////////////////
/// ProjectiveDecisionTreeLearner
////////////////////////////////////////////////////////////////////////////////


ProjectiveDecisionTree::ptr ProjectiveDecisionTreeLearner::learn(AbstractDataStorage::ptr dataStorage)
{
    AbstractDataStorage::ptr storage;
    // If we use bootstrap sampling, then this array contains the results of 
    // the sampler. We use it later in order to refine the leaf node histograms
    std::vector<bool> sampled;
    
    if (useBootstrap)
    {
        storage = dataStorage->bootstrap(numBootstrapExamples, sampled);
    }
    else
    {
        storage = dataStorage;
    }
    
    // Get the number of training examples and the dimensionality of the data set
    const int D = storage->getDimensionality();
    const int C = storage->getClasscount();
    
    // Set up a new tree. 
    ProjectiveDecisionTree::ptr tree = std::make_shared<ProjectiveDecisionTree>();
    tree->addNode();
    
    // Set up the state for the call backs
    DecisionTreeLearnerState state;
    state.action = ACTION_START_TREE;
    
    evokeCallback(tree, 0, state);
    
    // This is the list of nodes that still have to be split
    std::stack<int> splitStack;
    //splitStack.reserve(static_cast<int>(fastlog2(storage->getSize())));
    
    // Add the root node to the list of nodes that still have to be split
    splitStack.push(0);
    
    // This matrix stores the training examples for certain nodes. 
    std::vector< int* > trainingExamples;
    std::vector< int > trainingExamplesSizes;
    trainingExamples.reserve(LIBF_GRAPH_BUFFER_SIZE);
    trainingExamplesSizes.reserve(LIBF_GRAPH_BUFFER_SIZE);
    
    // Saves the sum of impurity decrease achieved by each feature
    importance = std::vector<float>(D, 0.f);
    
    // Add all training example to the root node
    trainingExamplesSizes.push_back(storage->getSize());
    trainingExamples.push_back(new int[trainingExamplesSizes[0]]);
    for (int n = 0; n < storage->getSize(); n++)
    {
        trainingExamples[0][n] = n;
    }
    
    // We use these arrays during training for the left and right histograms
    EfficientEntropyHistogram leftHistogram(C);
    EfficientEntropyHistogram rightHistogram(C);
    
    // Set up a probability distribution over the features
    std::mt19937 g(rd());
    std::uniform_int_distribution<int> dist(0, D-1);
    std::uniform_real_distribution<float> dist2(0, 1);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> dist3(-1, 1);
    const float s = std::sqrt(D);
    std::vector<float> projectionValues(storage->getSize(), 0.0f);
    
    int numProjections = numFeatures;
    // Set up some random projections
    std::vector<DataPoint> projections(numProjections);
    
    // Start training
    while (splitStack.size() > 0)
    {
        for (int f = 0; f < numProjections; f++)
        {
            projections[f] = DataPoint::Zero(D);
#if 1
                for (int d = 0; d < D; d++)
                {
                    const float u = dist2(g);
                    if (u <= 0.5/s)
                    {
                        projections[f](d) = -1;
                    }
                    else if ( u <= 1/s)
                    {
                        projections[f](d) = 1;
                    }
                }
#endif
#if 0
                projections[f](dist(g)) = 1;
                projections[f](dist(g)) = -1;
#endif
#if 0
                for (int d = 0; d < D; d++)
                {
                    projections[f](d) = normal(g);
                }
#endif
#if 0
                projections[f](dist(g)) = 1;
#endif
        }
        // Extract an element from the queue
        const int node = splitStack.top();
        splitStack.pop();
        
        // Get the training example list
        int* trainingExampleList = trainingExamples[node];
        const int N = trainingExamplesSizes[node];

        // Set up the right histogram
        // Because we start with the threshold being at the left most position
        // The right child node contains all training examples
        
        EfficientEntropyHistogram hist(C);
        for (int m = 0; m < N; m++)
        {
            // Get the class label of this training example
            hist.addOne(storage->getClassLabel(trainingExampleList[m]));
        }

        // Don't split this node
        //  If the number of examples is too small
        //  If the training examples are all of the same class
        //  If the maximum depth is reached
        if (hist.getMass() < minSplitExamples || hist.isPure() || tree->getNodeConfig(node).getDepth() >= maxDepth)
        {
            // Resize and initialize the leaf node histogram
            updateLeafNodeHistogram(tree->getNodeData(node).histogram, hist, smoothingParameter, useBootstrap);
            BOOST_ASSERT(tree->getNodeData(node).histogram.size() > 0);
            continue;
        }
        
        // These are the parameters we optimize
        float bestThreshold = 0;
        float bestObjective = 1e35;
        int bestLeftMass = 0;
        int bestRightMass = N;
        DataPoint bestProjection(D);

        // Optimize over all features
        for (int f = 0; f < numFeatures; f++)
        {
            // Set up the array of projection values
            for (int m = 0; m < N; m++)
            {
                const int n = trainingExampleList[m];
                projectionValues[n] = projections[f].adjoint()*storage->getDataPoint(n);
            }
            
            std::sort(trainingExampleList, trainingExampleList + N, [&projectionValues](const int lhs, const int rhs) -> bool {
                return projectionValues[lhs] < projectionValues[rhs];
            });
            
            // Initialize the histograms
            leftHistogram.reset();
            rightHistogram = hist;
            
            float leftValue = projectionValues[trainingExampleList[0]];
            int leftClass = storage->getClassLabel(trainingExampleList[0]);
            
            // Test different thresholds
            // Go over all examples in this node
            for (int m = 1; m < N; m++)
            {
                const int n = trainingExampleList[m];
                
                // Move the last point to the left histogram
                leftHistogram.addOne(leftClass);
                rightHistogram.subOne(leftClass);
                        
                // It does
                // Get the two feature values
                const float rightValue = projectionValues[n];
                
                // Skip this split, if the two points lie too close together
                const float diff = rightValue - leftValue;
                
                if (diff < 1e-6f)
                {
                    leftValue = rightValue;
                    leftClass = storage->getClassLabel(n);
                    continue;
                }
                
                // Get the objective function
                const float localObjective = leftHistogram.getEntropy()
                        + rightHistogram.getEntropy();
                
                if (localObjective < bestObjective)
                {
                    // Get the threshold value
                    bestThreshold = (leftValue + rightValue);
                    bestProjection = projections[f];
                    bestObjective = localObjective;
                    bestLeftMass = leftHistogram.getMass();
                    bestRightMass = rightHistogram.getMass();
                }
                
                leftValue = rightValue;
                leftClass = storage->getClassLabel(n);
            }
        }
        
        // We spare the additional multiplication at each iteration.
        bestThreshold *= 0.5f;
        
        // Did we find good split values?
        if (bestObjective > 1e20 || bestLeftMass < minChildSplitExamples || bestRightMass < minChildSplitExamples)
        {
            // We didn't
            // Don't split
            updateLeafNodeHistogram(tree->getNodeData(node).histogram, hist, smoothingParameter, useBootstrap);
            BOOST_ASSERT(tree->getNodeData(node).histogram.size() > 0);
            continue;
        }
        
        // Set up the data lists for the child nodes
        trainingExamplesSizes.push_back(bestLeftMass);
        trainingExamplesSizes.push_back(bestRightMass);
        trainingExamples.push_back(new int[bestLeftMass]);
        trainingExamples.push_back(new int[bestRightMass]);
        
        int* leftList = trainingExamples[trainingExamples.size() - 2];
        int* rightList = trainingExamples[trainingExamples.size() - 1];
        
        // Sort the points
        for (int m = 0; m < N; m++)
        {
            const int n = trainingExampleList[m];
            const float featureValue = bestProjection.adjoint()*storage->getDataPoint(n);
            
            if (featureValue < bestThreshold)
            {
                leftList[--bestLeftMass] = n;
            }
            else
            {
                rightList[--bestRightMass] = n;
            }
        }
        
        // Ok, split the node
        tree->getNodeConfig(node).setThreshold(bestThreshold);
        tree->getNodeConfig(node).getProjection() = bestProjection;
        const int leftChild = tree->splitNode(node);
        
        state.action = ACTION_SPLIT_NODE;
        state.depth = tree->getNodeConfig(node).getDepth();
        state.objective = bestObjective;
        
        evokeCallback(tree, 0, state);
        
        // Prepare to split the child nodes
        splitStack.push(leftChild);
        splitStack.push(leftChild + 1);
        
        delete[] trainingExampleList;
    }
    
    // If we use bootstrap, we use all the training examples for the 
    // histograms
    if (useBootstrap)
    {
        updateHistograms(tree, dataStorage);
    }
    
    return tree;
}

void ProjectiveDecisionTreeLearner::updateHistograms(ProjectiveDecisionTree::ptr tree, AbstractDataStorage::ptr storage) const
{
    const int C = storage->getClasscount();
    
    // Reset all histograms
    for (int v = 0; v < tree->getNumNodes(); v++)
    {
        if (tree->getNodeConfig(v).isLeafNode())
        {
            std::vector<float> & hist = tree->getNodeData(v).histogram;
            
            // Make sure that hist is initialized.
            hist.resize(C);
            
            for (int c = 0; c < C; c++)
            {
                hist[c] = 0;
            }
        }
    }
    
    
    // Compute the weights for each data point
    for (int n = 0; n < storage->getSize(); n++)
    {
        int leafNode = tree->findLeafNode(storage->getDataPoint(n));
        tree->getNodeData(leafNode).histogram[storage->getClassLabel(n)] += 1;
    }
    
    // Normalize the histograms
    for (int v = 0; v < tree->getNumNodes(); v++)
    {
        if (tree->getNodeConfig(v).isLeafNode())
        {
            std::vector<float> & hist = tree->getNodeData(v).histogram;
            float total = 0;
            for (int c = 0; c < C; c++)
            {
                total += hist[c];
            }
            for (int c = 0; c < C; c++)
            {
                hist[c] = std::log((hist[c] + smoothingParameter)/(total + C*smoothingParameter));
            }
        }
    }
}

int ProjectiveDecisionTreeLearner::defaultCallback(ProjectiveDecisionTree::ptr tree, const DecisionTreeLearnerState & state)
{
    switch (state.action) {
        case DecisionTreeLearner::ACTION_START_TREE:
            std::cout << "Start decision tree training" << "\n";
            break;
        case DecisionTreeLearner::ACTION_SPLIT_NODE:
            std::cout << std::setw(15) << std::left << "Split node:"
                    << "depth = " << std::setw(3) << std::right << state.depth
                    << ", objective = " << std::setw(6) << std::left
                    << std::setprecision(4) << state.objective << "\n";
            break;
        default:
            std::cout << "UNKNOWN ACTION CODE " << state.action << "\n";
            break;
    }
    
    return 0;

}
