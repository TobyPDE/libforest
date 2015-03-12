#include "libforest/unsupervised_learning.h"
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
/// DensityTreeLearner
////////////////////////////////////////////////////////////////////////////////

void DensityTreeLearner::updateLeafNodeGaussian(Gaussian & gaussian, EfficientCovarianceMatrix & covariance)
{
    gaussian.setMean(covariance.getMean());
    gaussian.setCovariance(covariance.getCovariance());

    assert(gaussian.getMean().rows() > 0);
    assert(gaussian.getCovariance().rows() > 0);
    assert(gaussian.getCovariance().cols() > 0);
}

DensityTree* DensityTreeLearner::learn(const UnlabeledDataStorage* storage)
{
    const int D = storage->getDimensionality();
    const int N = storage->getSize();
    
    numFeatures = std::min(D, numFeatures);
    
    // Set up a new density tree. 
    DensityTree* tree = new DensityTree();
    
    // Set up the state for the callbacks.
    DensityTreeLearnerState state;
    state.action = ACTION_START_TREE;
    
    evokeCallback(tree, 0, &state);
    
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
    
    std::vector<int> features(N);
    for (int f = 0; f < D; f++) 
    {
        features[f] = f;
    }

    while (splitStack.size() > 0)
    {
        // Extract an element from the queue
        const int leaf = splitStack.back();
        splitStack.pop_back();

        const int depth = tree->getDepth(leaf);
        
        state.action = ACTION_PROCESS_NODE;
        state.node = leaf;
        state.depth = depth;
                
        evokeCallback(tree, 0, &state);
        
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
        
        assert(N_leaf > 0);
        assert(covariance.getMass() == N_leaf);
        
        state.action = ACTION_INIT_NODE;
        evokeCallback(tree, 0, &state);
        
        // Don't split this node
        //  If the number of examples is too small
        //  If the training examples are all of the same class
        //  If the maximum depth is reached
        if (covariance.getMass() < minSplitExamples || tree->getDepth(leaf) > maxDepth)
        {
            state.action = ACTION_NOT_SPLIT_NODE;
            state.maxDepth = maxDepth;
            
            evokeCallback(tree, 0, &state);
        
            trainingExamples[leaf].clear();
            
            // Resize and initialize the leaf node histogram.
            updateLeafNodeGaussian(tree->getGaussian(leaf), covariance);
            
            continue;
        }
        
        // These are the parameters we optimize
        float bestThreshold = 0;
        int bestFeature = -1;
        float bestObjective = 1e35;

        // Sample random features
        std::shuffle(features.begin(), features.end(), std::default_random_engine(rd()));
        
        // Optimize over all features
        for (int f = 0; f < numFeatures; f++)
        {
            const int feature = features[f];
            
            cp.feature = feature;
            std::sort(trainingExamples[leaf].begin(), trainingExamples[leaf].end(), cp);
            
            leftCovariance.reset();
            rightCovariance = covariance;
            
            // Initialize left feature value.
            float leftFeatureValue = storage->getDataPoint(trainingExamples[leaf][0])->at(feature);
            
            // The training samples are our thresholds to optimize over.
            for (int m = 1; m < N_leaf; m++)
            {
                const int n = trainingExamples[leaf][m];
                
                // Shift threshold one sample to the right.
                leftCovariance.addOne(storage->getDataPoint(n));
                rightCovariance.subOne(storage->getDataPoint(n));
                
                float rightFeatureValue = storage->getDataPoint(n)->at(feature);
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
                    // Get the objective function
                    const float localObjective = leftCovariance.getEntropy()/N_leaf
                            + rightCovariance.getEntropy()/N_leaf;
                    
                    if (localObjective < bestObjective)
                    {std::cout << f << " " << m << " " << N_leaf << std::endl;
                        // Get the threshold value
                        bestThreshold = (leftFeatureValue + rightFeatureValue)/2;
                        bestFeature = feature;
                        bestObjective = localObjective;
                    }
                }
                
                leftFeatureValue = rightFeatureValue;
            }
        }
        
        // Did we find good split values?
        if (bestFeature < 0)
        {
            state.action = ACTION_NO_SPLIT_NODE;
            state.objective = bestObjective;
            
            evokeCallback(tree, 0, &state);
            
            // Don't split
            trainingExamples[leaf].clear();
           
            updateLeafNodeGaussian(tree->getGaussian(leaf), covariance);
            
            continue;
        }
        
        // Ok, split the node
        tree->setThreshold(leaf, bestThreshold);
        tree->setSplitFeature(leaf, bestFeature);
        
        const int leftChild = tree->splitNode(leaf);
        const int rightChild = leftChild + 1;
        
        // Set up the data lists for the child nodes
        trainingExamples.push_back(std::vector<int>());
        trainingExamples.push_back(std::vector<int>());
        
        std::vector<int> & leftTrainingExamples = trainingExamples[leftChild];
        std::vector<int> & rightTrainingExamples = trainingExamples[rightChild];
        
        // Sort the points
        for (int m = 0; m < N_leaf; m++)
        {
            const int n = trainingExamples[leaf][m];
            assert(n >= 0 && n < storage->getSize());
            const float featureValue = storage->getDataPoint(n)->at(bestFeature);
            
            if (featureValue < bestThreshold)
            {
                leftTrainingExamples.push_back(n);
            }
            else
            {
                rightTrainingExamples.push_back(n);
            }
        }
        
        assert(leftTrainingExamples.size() > 0);
        assert(rightTrainingExamples.size() > 0);
        
        state.action = ACTION_SPLIT_NODE;
        state.depth = depth;
        state.objective = bestObjective;
        
        evokeCallback(tree, 0, &state);
        
        // Prepare to split the child nodes
        splitStack.push_back(leftChild);
        splitStack.push_back(rightChild);
    }
    
    return tree;
}

int DensityTreeLearner::defaultCallback(DensityTree* tree, DensityTreeLearnerState* state)
{
    switch (state->action) {
        case DensityTreeLearner::ACTION_START_TREE:
            std::cout << "Start decision tree training" << "\n";
            break;
        case DensityTreeLearner::ACTION_SPLIT_NODE:
            std::cout << std::setw(15) << std::left << "Split node:"
                    << "depth = " << std::setw(3) << std::right << state->depth
                    << ", objective = " << std::setw(6) << std::left
                    << std::setprecision(4) << state->objective << "\n";
            break;
        case DensityTreeLearner::ACTION_INIT_NODE:
            std::cout << std::setw(15) << std::left << "Init node:"
                    << "depth = " << std::setw(3) << std::right << state->depth << "\n";
            break;
        case DensityTreeLearner::ACTION_NOT_SPLIT_NODE:
            std::cout << std::setw(15) << std::left << "Not split node:"
                    << "depth = " << std::setw(3) << std::right << state->depth
                    << ", max depth = " << std::setw(3) << std::right << state->maxDepth << "\n";
            break;
        case DensityTreeLearner::ACTION_NO_SPLIT_NODE:
            std::cout << std::setw(15) << std::left << "Not split node:"
                    << "depth = " << std::setw(3) << std::right << state->depth
                    << ", objective = " << std::setw(6) << std::left
                    << std::setprecision(4) << state->objective << "\n";
            break;
        case DensityTreeLearner::ACTION_PROCESS_NODE:
            std::cout << std::setw(15) << std::left << "Process node:"
                    << "depth = " << std::setw(3) << std::right << state->depth << "\n";
            break;
        default:
            std::cout << "UNKNOWN ACTION CODE " << state->action << "\n";
            break;
    }
    return 0;
}