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
/// DensityDecisionTreeLearner
////////////////////////////////////////////////////////////////////////////////

void DensityDecisionTreeLearner::updateLeafNodeGaussian(Gaussian gaussian, EfficientCovarianceMatrix covariance)
{
    gaussian = Gaussian(covariance.getMean(), covariance.getCovariance(), covariance.getDeterminant());
}

DensityDecisionTree* DensityDecisionTreeLearner::learn(const UnlabeledDataStorage* storage)
{
    const int D = storage->getDimensionality();
    const int N = storage->getSize();
    
    // Set up a new density tree. 
    DensityDecisionTree* tree = new DensityDecisionTree();
    
    // Set up the state for the callbacks.
    DensityDecisionTreeLearnerState state;
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

        const int N_leaf = trainingExamples[leaf].size();
        
        // Set up the right histogram
        // Because we start with the threshold being at the left most position
        // The right child node contains all training examples
        
        EfficientCovarianceMatrix covariance(D);
        for (int m = 0; m < N_leaf; m++)
        {
            covariance.addOne(storage->getDataPoint(m));
        }

        // Don't split this node
        //  If the number of examples is too small
        //  If the training examples are all of the same class
        //  If the maximum depth is reached
        if (covariance.getMass() < minSplitExamples || tree->getDepth(leaf) > maxDepth)
        {
            trainingExamples[leaf].clear();
            // Resize and initialize the leaf node histogram
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
            for (int m = 1; m < N; m++)
            {
                const int n = trainingExamples[leaf][m];
                
                // Shift threshold one sample to the right.
                leftCovariance.addOne(storage->getDataPoint(n));
                rightCovariance.subOne(storage->getDataPoint(n));
                
                const float rightFeatureValue = storage->getDataPoint(n)->at(feature);
                if (std::abs(rightFeatureValue - leftFeatureValue) < 1e-6f)
                {
                    leftFeatureValue = rightFeatureValue;
                    continue;
                }
                
                // Only try if enough samples would be in the new children.
                if (leftCovariance.getMass() > minChildSplitExamples
                        && rightCovariance.getMass() > minChildSplitExamples)
                {
                    // Get the objective function
                    const float localObjective = leftCovariance.getEntropy()
                            + rightCovariance.getEntropy();

                    if (localObjective < bestObjective)
                    {
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
            // Don't split
            trainingExamples[leaf].clear();
            updateLeafNodeGaussian(tree->getGaussian(leaf), covariance);
            continue;
        }
        
        // Set up the data lists for the child nodes
        trainingExamples.push_back(std::vector<int>());
        trainingExamples.push_back(std::vector<int>());
        
        std::vector<int> leftTrainingExamples = trainingExamples[trainingExamples.size() - 2];
        std::vector<int> rightTrainingExamples = trainingExamples[trainingExamples.size() - 1];
        
        // Sort the points
        for (int m = 0; m < N; m++)
        {
            const int n = trainingExamples[leaf][m];
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
        
        // Ok, split the node
        tree->setThreshold(leaf, bestThreshold);
        tree->setSplitFeature(leaf, bestFeature);
        const int leftChild = tree->splitNode(leaf);
        
        state.action = ACTION_SPLIT_NODE;
        state.depth = tree->getDepth(leaf);
        state.objective = bestObjective;
        
        evokeCallback(tree, 0, &state);
        
        // Prepare to split the child nodes
        splitStack.push_back(leftChild);
        splitStack.push_back(leftChild + 1);
    }
    
    return tree;
}

int DensityDecisionTreeLearner::defaultCallback(DensityDecisionTree* tree, DensityDecisionTreeLearnerState* state)
{
    switch (state->action) {
        case DecisionTreeLearner::ACTION_START_TREE:
            std::cout << "Start decision tree training" << "\n";
            break;
        case DecisionTreeLearner::ACTION_SPLIT_NODE:
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