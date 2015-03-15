#include "libforest/learning.h"
#include "libforest/data.h"
#include "libforest/classifiers.h"
#include "libforest/util.h"

#include <algorithm>
#include <random>
#include <map>
#include <iomanip>

using namespace libf;

static std::random_device rd;

////////////////////////////////////////////////////////////////////////////////
/// DecisionTreeLearner
////////////////////////////////////////////////////////////////////////////////

/**
 * Updates the leaf node histograms using a smoothing parameter
 */
inline void updateLeafNodeHistogram(std::vector<float> & leafNodeHistograms, const EfficientEntropyHistogram & hist, float smoothing, bool useBootstrap)
{
    leafNodeHistograms.resize(hist.getSize());
    
    if(!useBootstrap)
    {
        for (int c = 0; c < hist.getSize(); c++)
        {
            leafNodeHistograms[c] = std::log((hist.at(c) + smoothing)/(hist.getMass() + hist.getSize() * smoothing));
        }
    }
}

DecisionTree::ptr DecisionTreeLearner::learn(AbstractDataStorage::ptr dataStorage)
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
    
    // Set up a new tree. Note: We will convert the tree to a shared pointer at
    // The end in order to speed up learning. 
    DecisionTree::ptr tree = std::make_shared<DecisionTree>();
    
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
        if (hist.getMass() < minSplitExamples || hist.isPure() || tree->getDepth(node) > maxDepth)
        {
            delete[] trainingExampleList;
            // Resize and initialize the leaf node histogram
            updateLeafNodeHistogram(tree->getHistogram(node), hist, smoothingParameter, useBootstrap);
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
            delete[] trainingExampleList;
            updateLeafNodeHistogram(tree->getHistogram(node), hist, smoothingParameter, useBootstrap);
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
        tree->setThreshold(node, bestThreshold);
        tree->setSplitFeature(node, bestFeature);
        const int leftChild = tree->splitNode(node);
        
        state.action = ACTION_SPLIT_NODE;
        state.depth = tree->getDepth(node);
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
        if (tree->isLeafNode(v))
        {
            std::vector<float> & hist = tree->getHistogram(v);
            
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
        tree->getHistogram(leafNode)[storage->getClassLabel(n)] += 1;
    }
    
    // Normalize the histograms
    for (int v = 0; v < tree->getNumNodes(); v++)
    {
        if (tree->isLeafNode(v))
        {
            std::vector<float> & hist = tree->getHistogram(v);
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
/// RandomForestLearner
////////////////////////////////////////////////////////////////////////////////

RandomForest::ptr RandomForestLearner::learn(AbstractDataStorage::ptr storage)
{
    // Set up the empty random forest
    RandomForest::ptr forest = std::make_shared<RandomForest>();
    
    const int D = storage->getDimensionality();
    
    // Initialize variable importance values.
    importance = std::vector<float>(D, 0.f);
    
    // Set up the state for the call backs
    RandomForestLearnerState state;
    state.numTrees = numTrees;
    state.tree = 0;
    state.action = ACTION_START_FOREST;
    
    evokeCallback(forest, 0, state);
    
    int treeStartCounter = 0; 
    int treeFinishCounter = 0; 
    #pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < numTrees; i++)
    {
        #pragma omp critical
        {
            state.tree = ++treeStartCounter;
            state.action = ACTION_START_TREE;
            evokeCallback(forest, treeStartCounter - 1, state);
        }
        
        // Learn the tree
        DecisionTree::ptr tree = treeLearner.learn(storage);
        // Add it to the forest
        #pragma omp critical
        {
            state.tree = ++treeFinishCounter;
            state.action = ACTION_FINISH_TREE;
            evokeCallback(forest, treeFinishCounter - 1, state);
            forest->addTree(tree);

            // Update variable importance.
            for (int f = 0; f < D; ++f)
            {
                importance[f] += treeLearner.getImportance(f)/this->numTrees;
            }
        }
    }
    
    state.tree = 0;
    state.action = ACTION_FINISH_FOREST;
    evokeCallback(forest, 0, state);
    
    return RandomForest::ptr(forest);
}

int RandomForestLearner::defaultCallback(RandomForest::ptr forest, const RandomForestLearnerState & state)
{
    switch (state.action) {
        case RandomForestLearner::ACTION_START_FOREST:
            std::cout << "Start random forest training" << "\n";
            break;
        case RandomForestLearner::ACTION_START_TREE:
            std::cout << std::setw(15) << std::left << "Start tree " 
                    << std::setw(4) << std::right << state.tree 
                    << " out of " 
                    << std::setw(4) << state.numTrees << "\n";
            break;
        case RandomForestLearner::ACTION_FINISH_TREE:
            std::cout << std::setw(15) << std::left << "Finish tree " 
                    << std::setw(4) << std::right << state.tree 
                    << " out of " 
                    << std::setw(4) << state.numTrees << "\n";
            break;
        case RandomForestLearner::ACTION_FINISH_FOREST:
            std::cout << "Finished forest in " << state.getPassedTime().count()/1000000. << "s\n";
            break;
        default:
            std::cout << "UNKNOWN ACTION CODE " << state.action << "\n";
            break;
    }
    
    return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// BoostedRandomForestLearner
////////////////////////////////////////////////////////////////////////////////

BoostedRandomForest::ptr BoostedRandomForestLearner::learn(AbstractDataStorage::ptr storage)
{
    // Set up the empty random forest
    BoostedRandomForest::ptr forest = std::make_shared<BoostedRandomForest>();
    
    // Set up the state for the call backs
    BoostedRandomForestLearnerState state;
    state.numTrees = this->getNumTrees();
    state.tree = 0;
    state.action = ACTION_START_FOREST;
    
    evokeCallback(forest, 0, state);
    
    // Set up the weights for the data points
    const int N = storage->getSize();
    std::vector<float> dataWeights(N);
    std::vector<float> cumsum(N);
    std::vector<bool> misclassified(N);
    for (int n = 0; n < N; n++)
    {
        dataWeights[n] = 1.0f/N;
        cumsum[n] = (n+1) * 1.0f/N;
        misclassified[n] = false;
    }
    
    // We need this distribution in order to sample according to the weights
    std::mt19937 g(rd());
    std::uniform_real_distribution<float> U(0, 1);
    
    const int C = storage->getClasscount();
    
    int treeStartCounter = 0; 
    int treeFinishCounter = 0; 
    for (int i = 0; i < numTrees; i++)
    {
        state.tree = ++treeStartCounter;
        state.action = ACTION_START_TREE;
        evokeCallback(forest, treeStartCounter - 1, state);
        
        // Learn the tree
        // --------------
        
        // Sample data points according to the weights
        ReferenceDataStorage::ptr treeData = std::make_shared<ReferenceDataStorage>(storage);
        
        for (int n = 0; n < N; n++)
        {
            const float u = U(g);
            int index = 0;
            while (u > cumsum[index] && index < N-1)
            {
                index++;
            }
            treeData->addDataPoint(index);
        }
        
        // Learn the tree
        DecisionTree::ptr tree = treeLearner.learn(treeData);
        
        // Calculate the error term
        float error = 0;
        for (int n = 0; n < N; n++)
        {
            const int predictedLabel = tree->classify(storage->getDataPoint(n));
            if (predictedLabel != storage->getClassLabel(n))
            {
                error += dataWeights[n];
                misclassified[n] = true;
            }
            else
            {
                misclassified[n] = false;
            }
        }
        
        // Compute the classifier weight
        const float alpha = std::log((1-error)/error) + std::log(C - 1);
        
        // Update the weights
        float total = 0;
        for (int n = 0; n < N; n++)
        {
            if (misclassified[n])
            {
                dataWeights[n] *= std::exp(alpha);
            }
            total += dataWeights[n];
        }
        dataWeights[0] /= total;
        cumsum[0] = dataWeights[0];
        for (int n = 1; n < N; n++)
        {
            dataWeights[n] /= total;
            cumsum[n] = dataWeights[n] + cumsum[n-1];
        }
        
        // Add the classifier
        forest->addTree(tree, alpha);
        
        // --------------
        // Add it to the forest
        state.tree = ++treeFinishCounter;
        state.error = error;
        state.alpha = alpha;
        state.action = ACTION_FINISH_TREE;
        evokeCallback(forest, treeFinishCounter - 1, state);
    }
    
    state.tree = 0;
    state.action = ACTION_FINISH_FOREST;
    evokeCallback(forest, 0, state);
    
    return forest;
}

int BoostedRandomForestLearner::defaultCallback(BoostedRandomForest::ptr forest, const BoostedRandomForestLearnerState & state)
{
    switch (state.action) {
        case BoostedRandomForestLearner::ACTION_START_FOREST:
            std::cout << "Start boosted random forest training\n" << "\n";
            break;
        case BoostedRandomForestLearner::ACTION_START_TREE:
            std::cout   << std::setw(15) << std::left << "Start tree " 
                        << std::setw(4) << std::right << state.tree 
                        << " out of " 
                        << std::setw(4) << state.numTrees << "\n";
            break;
        case BoostedRandomForestLearner::ACTION_FINISH_TREE:
            std::cout   << std::setw(15) << std::left << "Finish tree " 
                        << std::setw(4) << std::right << state.tree 
                        << " out of " 
                        << std::setw(4) << state.numTrees
                        << " error = " << state.error 
                        << ", alpha = " << state.alpha << "\n";
            break;
        case BoostedRandomForestLearner::ACTION_FINISH_FOREST:
            std::cout << "Finished boosted forest in " << state.getPassedTime().count()/1000000. << "s\n";
            break;
        default:
            std::cout << "UNKNOWN ACTION CODE " << state.action << "\n";
            break;
    }
    
    return 0;
}
