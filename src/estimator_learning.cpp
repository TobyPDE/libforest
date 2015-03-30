#include "libforest/estimator_learning.h"
#include "libforest/data.h"
#include "libforest/util.h"
#include "libforest/learning_tools.h"

#include <algorithm>
#include <random>
#include <map>
#include <iomanip>

using namespace libf;

static std::random_device rd;
static std::mt19937 g(rd());

////////////////////////////////////////////////////////////////////////////////
/// DensityTreeLearner
////////////////////////////////////////////////////////////////////////////////

void DensityTreeLearner::updateLeafNodeGaussian(Gaussian & gaussian, EfficientCovarianceMatrix & covariance)
{
    gaussian.setMean(covariance.getMean());
    gaussian.setCovariance(covariance.getCovariance());
    gaussian.setDataSupport(covariance.getMass());
    
    assert(gaussian.getMean().rows() > 0);
    assert(gaussian.getCovariance().rows() > 0);
    assert(gaussian.getCovariance().cols() > 0);
}

DensityTree::ptr DensityTreeLearner::learn(AbstractDataStorage::ptr storage, DensityTreeLearner::State & state)
{
    state.reset();
    state.started = true;
    
    const int D = storage->getDimensionality();
    const int N = storage->getSize();
    
    numFeatures = std::min(D, numFeatures);
    
    // Set up a new density tree. 
    DensityTree::ptr tree = std::make_shared<DensityTree>();
    tree->addNode();
    
    state.total = storage->getSize();
    
    // Nodes we need to split.
    std::vector<int> splitStack;
    splitStack.reserve(static_cast<int>(fastlog2(storage->getSize())));
    splitStack.push_back(0);
    
    // This matrix stores the training examples for certain nodes. 
    std::vector< std::vector<int> > trainingExamples;
    trainingExamples.reserve(LIBF_GRAPH_BUFFER_SIZE);
    
    // Add all training example to the root node
    trainingExamples.push_back(std::vector<int>(N));
    for (int n = 0; n < N; n++)
    {
        trainingExamples[0][n] = n;
    }
    
    EfficientCovarianceMatrix covariance(D);
    EfficientCovarianceMatrix leftCovariance(D);
    EfficientCovarianceMatrix rightCovariance(D);
    
    // We use this in order to sort the data points
    FeatureComparator cp;
    cp.storage = storage;
    
    std::vector<int> features(D);
    for (int f = 0; f < D; f++) 
    {
        features[f] = f;
    }

    while (splitStack.size() > 0)
    {
        // Extract an element from the queue
        const int leaf = splitStack.back();
        splitStack.pop_back();
        
        state.numNodes++;
        state.depth = std::max(state.depth, tree->getNodeConfig(leaf).getDepth());
        
        const int N_leaf = trainingExamples[leaf].size();
        
        // Set up the right histogram
        // Because we start with the threshold being at the left most position
        // The right child node contains all training examples
        
        covariance.reset();
        for (int m = 0; m < N_leaf; m++)
        {
            // Add all training examples of this leaf node.
            const int n = trainingExamples[leaf][m];
            covariance.addOne(storage->getDataPoint(n));
        }
        
        BOOST_ASSERT(N_leaf > 0);
        BOOST_ASSERT(covariance.getMass() == N_leaf);
        
        // Don't split this node
        //  If the number of examples is too small
        //  If the training examples are all of the same class
        //  If the maximum depth is reached
        if (covariance.getMass() < minSplitExamples || tree->getNodeConfig(leaf).getDepth() >= maxDepth)
        {
            // Resize and initialize the leaf node histogram.
            updateLeafNodeGaussian(tree->getNodeData(leaf).gaussian, covariance);
            
            trainingExamples[leaf].clear();
            state.processed += N_leaf;
            continue;
        }
        
        // These are the parameters we optimize
        float bestThreshold = 0;
        int bestFeature = -1;
        float bestObjective = 1e35;

        // Sample random features
        std::shuffle(features.begin(), features.end(), std::default_random_engine(rd()));
        
        // Optimize over all features
        for (int f = 1; f < numFeatures; f++)
        {
            const int feature = features[f];
            
            cp.feature = feature;
            std::sort(trainingExamples[leaf].begin(), trainingExamples[leaf].end(), cp);
            
            leftCovariance.reset();
            rightCovariance = covariance;
            
            // Initialize left feature value.
            float leftFeatureValue = storage->getDataPoint(trainingExamples[leaf][0])(feature);
            
            // The training samples are our thresholds to optimize over.
            for (int m = 1; m < N_leaf - 1; m++)
            {
                const int n = trainingExamples[leaf][m];
                
                // Shift threshold one sample to the right.
                leftCovariance.addOne(storage->getDataPoint(n));
                rightCovariance.subOne(storage->getDataPoint(n));
                assert(leftCovariance.getMass() + rightCovariance.getMass() == covariance.getMass());
                
                const float rightFeatureValue = storage->getDataPoint(n)(feature);
                assert(rightFeatureValue >= leftFeatureValue);
                
                if (rightFeatureValue - leftFeatureValue < 1e-6f)
                {
                    leftFeatureValue = rightFeatureValue;
                    continue;
                }
                
                // Only try if enough samples would be in the new children.
                if (leftCovariance.getMass() > minChildSplitExamples
                        && rightCovariance.getMass() > minChildSplitExamples)
                {
                    // Get the objective value.
                    const float localObjective = leftCovariance.getEntropy()/N_leaf
                            + rightCovariance.getEntropy()/N_leaf;
                    
                    if (localObjective < bestObjective)
                    {
                        bestThreshold = (leftFeatureValue + rightFeatureValue)/2;
                        bestFeature = feature;
                        bestObjective = localObjective;
                    }
                }
                
                leftFeatureValue = rightFeatureValue;
            }
        }
        
        // Did we find good split values?
        if (bestFeature < 0 && bestObjective >= 10)
        {
            // Don't split
            updateLeafNodeGaussian(tree->getNodeData(leaf).gaussian, covariance);
            
            trainingExamples[leaf].clear();
            state.processed += N_leaf;
            continue;
        }

        // Ok, split the node
        tree->getNodeConfig(leaf).setThreshold(bestThreshold);
        tree->getNodeConfig(leaf).setSplitFeature(bestFeature);
        
        const int leftChild = tree->splitNode(leaf);
        const int rightChild = leftChild + 1;
        
        // Set up the data lists for the child nodes
        trainingExamples.push_back(std::vector<int>());
        trainingExamples.push_back(std::vector<int>());
        
        // Sort the points
        for (int m = 0; m < N_leaf; m++)
        {
            const int n = trainingExamples[leaf][m];
            assert(n >= 0 && n < storage->getSize());
            
            const float featureValue = storage->getDataPoint(n)(bestFeature);
            
            if (featureValue < bestThreshold)
            {
                trainingExamples[leftChild].push_back(n);
            }
            else
            {
                trainingExamples[rightChild].push_back(n);
            }
        }
        
        // Prepare to split the child nodes
        splitStack.push_back(leftChild);
        splitStack.push_back(rightChild);
    }
    
    state.terminated = true;
    
    return tree;
}

////////////////////////////////////////////////////////////////////////////////
/// KernelDensityTreeLearner
////////////////////////////////////////////////////////////////////////////////

void KernelDensityTreeLearner::initializeLeafNodeEstimator(Gaussian & gaussian, 
        EfficientCovarianceMatrix & covariance,
        KernelDensityEstimator & estimator,
        const std::vector<int> & trainingExamples, 
        const AbstractDataStorage::ptr storage)
{
    const int N = static_cast<int>(trainingExamples.size());
    DataStorage::ptr leafStorage = DataStorage::Factory::create();
    
    for (int n = 0; n < N; n++)
    {
        leafStorage->addDataPoint(storage->getDataPoint(trainingExamples[n]));
    }
    
    estimator.setKernel(kernel);
    estimator.setDataStorage(leafStorage);
    estimator.selectBandwidth(bandwidthSelectionMethod);
    
    gaussian.setMean(covariance.getMean());
    gaussian.setCovariance(covariance.getCovariance());
    gaussian.setDataSupport(covariance.getMass());
    
    assert(gaussian.getMean().rows() > 0);
    assert(gaussian.getCovariance().rows() > 0);
    assert(gaussian.getDataSupport() > 0);
    assert(gaussian.getDataSupport() == leafStorage->getSize());
}

KernelDensityTree::ptr KernelDensityTreeLearner::learn(AbstractDataStorage::ptr storage, KernelDensityTreeLearner::State & state)
{
    const int D = storage->getDimensionality();
    const int N = storage->getSize();
    
    state.reset();
    state.started = true;
    state.total = N;
    
    numFeatures = std::min(D, numFeatures);
    
    // Set up a new density tree. 
    KernelDensityTree::ptr tree = std::make_shared<KernelDensityTree>();
    tree->addNode();
    
    // Nodes we need to split.
    std::vector<int> splitStack;
    splitStack.reserve(static_cast<int>(fastlog2(storage->getSize())));
    splitStack.push_back(0);
    
    // This matrix stores the training examples for certain nodes. 
    std::vector< std::vector<int> > trainingExamples;
    trainingExamples.reserve(LIBF_GRAPH_BUFFER_SIZE);
    
    // Add all training example to the root node
    trainingExamples.push_back(std::vector<int>(N));
    for (int n = 0; n < N; n++)
    {
        trainingExamples[0][n] = n;
    }
    
    EfficientCovarianceMatrix covariance(D);
    EfficientCovarianceMatrix leftCovariance(D);
    EfficientCovarianceMatrix rightCovariance(D);
    
    // We use this in order to sort the data points
    FeatureComparator cp;
    cp.storage = storage;
    
    std::vector<int> features(D);
    for (int f = 0; f < D; f++) 
    {
        features[f] = f;
    }

    while (splitStack.size() > 0)
    {
        // Extract an element from the queue
        const int leaf = splitStack.back();
        splitStack.pop_back();

        state.numNodes++;
        state.depth = std::max(state.depth, tree->getNodeConfig(leaf).getDepth());
        
        const int N_leaf = trainingExamples[leaf].size();
        
        // Set up the right histogram
        // Because we start with the threshold being at the left most position
        // The right child node contains all training examples
        
        covariance.reset();
        for (int m = 0; m < N_leaf; m++)
        {
            // Add all training examples of this leaf node.
            const int n = trainingExamples[leaf][m];
            covariance.addOne(storage->getDataPoint(n));
        }
        
        BOOST_ASSERT(N_leaf > 0);
        BOOST_ASSERT(covariance.getMass() == N_leaf);
        
        // Don't split this node
        //  If the number of examples is too small
        //  If the training examples are all of the same class
        //  If the maximum depth is reached
        if (covariance.getMass() < minSplitExamples || tree->getNodeConfig(leaf).getDepth() >= maxDepth)
        {
            // Initialize the kernel density estimator at the leaf node.
            initializeLeafNodeEstimator(tree->getNodeData(leaf).gaussian, covariance,
                    tree->getNodeData(leaf).estimator, trainingExamples[leaf], storage);
            
            trainingExamples[leaf].clear();
            state.processed += N_leaf;
            continue;
        }
        
        // These are the parameters we optimize
        float bestThreshold = 0;
        int bestFeature = -1;
        float bestObjective = 1e35;

        // Sample random features
        std::shuffle(features.begin(), features.end(), std::default_random_engine(rd()));
        
        // Optimize over all features
        for (int f = 1; f < numFeatures; f++)
        {
            const int feature = features[f];
            
            cp.feature = feature;
            std::sort(trainingExamples[leaf].begin(), trainingExamples[leaf].end(), cp);
            
            leftCovariance.reset();
            rightCovariance = covariance;
            
            // Initialize left feature value.
            float leftFeatureValue = storage->getDataPoint(trainingExamples[leaf][0])(feature);
            
            // The training samples are our thresholds to optimize over.
            for (int m = 1; m < N_leaf - 1; m++)
            {
                const int n = trainingExamples[leaf][m];
                
                // Shift threshold one sample to the right.
                leftCovariance.addOne(storage->getDataPoint(n));
                rightCovariance.subOne(storage->getDataPoint(n));
                assert(leftCovariance.getMass() + rightCovariance.getMass() == covariance.getMass());
                
                float rightFeatureValue = storage->getDataPoint(n)(feature);
                assert(rightFeatureValue >= leftFeatureValue);
                
                if (rightFeatureValue - leftFeatureValue < 1e-6f)
                {
                    leftFeatureValue = rightFeatureValue;
                    continue;
                }
                
                // Only try if enough samples would be in the new children.
                if (leftCovariance.getMass() > minChildSplitExamples
                        && rightCovariance.getMass() > minChildSplitExamples)
                {
                    // Get the objective value.
                    const float localObjective = leftCovariance.getEntropy()/N_leaf
                            + rightCovariance.getEntropy()/N_leaf;
                    
                    if (localObjective < bestObjective)
                    {
                        bestThreshold = (leftFeatureValue + rightFeatureValue)/2;
                        bestFeature = feature;
                        bestObjective = localObjective;
                    }
                }
                
                leftFeatureValue = rightFeatureValue;
            }
        }
        
        // Did we find good split values?
        if (bestFeature < 0 && bestObjective >= 10)
        {
            // Initialize the kernel density estimator at the leaf node.
            initializeLeafNodeEstimator(tree->getNodeData(leaf).gaussian, covariance,
                    tree->getNodeData(leaf).estimator, trainingExamples[leaf], storage);
            
            trainingExamples[leaf].clear();
            state.processed += N_leaf;
            continue;
        }
        
        // Ok, split the node
        tree->getNodeConfig(leaf).setThreshold(bestThreshold);
        tree->getNodeConfig(leaf).setSplitFeature(bestFeature);
        
        const int leftChild = tree->splitNode(leaf);
        const int rightChild = leftChild + 1;
        
        // Set up the data lists for the child nodes
        trainingExamples.push_back(std::vector<int>());
        trainingExamples.push_back(std::vector<int>());
        
        // Sort the points
        for (int m = 0; m < N_leaf; m++)
        {
            const int n = trainingExamples[leaf][m];
            assert(n >= 0 && n < storage->getSize());
            
            const float featureValue = storage->getDataPoint(n)(bestFeature);
            
            if (featureValue < bestThreshold)
            {
                trainingExamples[leftChild].push_back(n);
            }
            else
            {
                trainingExamples[rightChild].push_back(n);
            }
        }
        
        // Prepare to split the child nodes
        splitStack.push_back(leftChild);
        splitStack.push_back(rightChild);
    }
    
    state.terminated = true;
    
    return tree;
}
