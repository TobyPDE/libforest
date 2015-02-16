#include "libforest/learning.h"
#include "libforest/data.h"
#include "libforest/classifiers.h"
#include "libforest/util.h"
#include "fastlog.h"
#include "mcmc.h"

#include <algorithm>
#include <random>
#include <map>
#include <iomanip>

using namespace libf;

#define ENTROPY(p) (-(p)*fastlog2(p))

static std::random_device rd;

////////////////////////////////////////////////////////////////////////////////
/// DecisionTreeLearner
////////////////////////////////////////////////////////////////////////////////

/**
 * A histogram over the class labels. We use this for training
 */
class EfficientEntropyHistogram {
private:
    /**
     * The number of classes in this histogram
     */
    unsigned char bins;

    /**
     * The actual histogram
     */
    int* histogram;

    /**
     * The integral over the entire histogram
     */
    int mass;

    /**
     * The entropies for the single bins
     */
    float* entropies;

    /**
     * The total entropy
     */
    float totalEntropy;

public:
    /**
     * Default constructor
     */
    EfficientEntropyHistogram() : bins(0), histogram(0), mass(0), entropies(0), totalEntropy(0) { }
    EfficientEntropyHistogram(int _classCount) : bins(_classCount), histogram(0), mass(0), entropies(0), totalEntropy(0) { resize(_classCount); }

    /**
     * Copy constructor
     */
    EfficientEntropyHistogram(const EfficientEntropyHistogram & other) 
    {
        resize (other.bins);
        for (int i = 0; i < bins; i++)
        {
            set(i, other.at(i));
        }
        mass = other.mass;
    }

    /**
     * Assignment operator
     */
    EfficientEntropyHistogram & operator= (const EfficientEntropyHistogram &other)
    {
        // Prevent self assignment
        if (this != &other)
        {
            if (other.bins != bins)
            {
                resize (other.bins);
            }
            for (int i = 0; i < bins; i++)
            {
                set(i, other.at(i));
                entropies[i] = other.entropies[i];
            }
            mass = other.mass;
            totalEntropy = other.totalEntropy;
        }
        return *this;
    }

    /**
     * Destructor
     */
    ~EfficientEntropyHistogram()
    {
        if (histogram != 0)
        {
            delete[] histogram;
        }
        if (entropies != 0)
        {
            delete[] entropies;
        }
    }

    /**
     * Resizes the histogram to a certain size
     */
    void resize(int _classCount)
    {
        // Release the current histogram
        if (histogram != 0)
        {
            delete[] histogram;
            histogram = 0;
        }
        if (entropies != 0)
        {
            delete[] entropies;
            entropies = 0;
        }

        // Only allocate a new histogram, if there is more than one class
        if (_classCount > 0)
        {
            histogram = new int[_classCount];
            entropies = new float[_classCount];
            bins = _classCount;

            // Initialize the histogram
            for (int i = 0; i < bins; i++)
            {
                histogram[i] = 0;
                entropies[i] = 0;
            }
        }
    }

    /**
     * Returns the size of the histogram (= class count)
     */
    int size() const { return bins; }

    /**
     * Returns the value of the histogram at a certain position. Caution: For performance reasons, we don't
     * perform any parameter check!
     */
    int at(const int i) const { return histogram[i]; }
    int get(const int i) const { return histogram[i]; }
    void set(const int i, const int v) { mass -= histogram[i]; mass += v; histogram[i] = v; }
    void add(const int i, const int v) { mass += v; histogram[i] += v; }
    void sub(const int i, const int v) { mass -= v; histogram[i] -= v; }
    void add1(const int i) { mass++; histogram[i]++; }
    void sub1(const int i) { mass--; histogram[i]--; }
    void addOne(const int i)
    {
        totalEntropy += ENTROPY(getMass());
        mass++;
        totalEntropy -= ENTROPY(getMass());
        histogram[i]++;
        totalEntropy -= entropies[i];
        entropies[i] = ENTROPY(histogram[i]); 
        totalEntropy += entropies[i];
    }
    void subOne(const int i)
    { 
        totalEntropy += ENTROPY(getMass());
        mass--; 
        totalEntropy -= ENTROPY(getMass());

        histogram[i]--;
        totalEntropy -= entropies[i];
        if (histogram[i] < 1)
        {
            entropies[i] = 0;
        }
        else
        {
            entropies[i] = ENTROPY(histogram[i]); 
            totalEntropy += entropies[i];
        }
    }

    /**
     * Returns the mass
     */
    float getMass() const
    {
        return mass;
    }

    /**
     * Calculates the entropy of a histogram
     * 
     * @return The calculated entropy
     */
    float entropy() const
    {
        return totalEntropy;
    }

    /**
     * Initializes all entropies
     */
    void initEntropies()
    {
        if (getMass() > 1)
        {
            totalEntropy = -ENTROPY(getMass());
            for (int i = 0; i < bins; i++)
            {
                if (at(i) == 0) continue;

                entropies[i] = ENTROPY(histogram[i]);

                totalEntropy += entropies[i];
            }
        }
    }

    /**
     * Sets all entries in the histogram to 0
     */
    void reset()
    {
        for (int i = 0; i < bins; i++)
        {
            histogram[i] = 0;
            entropies[i] = 0;
        }
        totalEntropy = 0;
        mass = 0;
    }
    
    /**
     * Returns the greatest bin
     */
    int argMax() const
    {
        int maxBin = 0;
        int maxCount = histogram[0];
        for (int i = 1; i < bins; i++)
        {
            if (histogram[i] > maxCount)
            {
                maxCount = at(i);
                maxBin = i;
            }
        }
        
        return maxBin;
    }
    
    /**
     * Returns true if the histogram is pure
     */
    bool isPure() const
    {
        bool nonPure = false;
        for (int i = 0; i < bins; i++)
        {
            if (histogram[i] > 0)
            {
                if (nonPure)
                {
                    return false;
                }
                else
                {
                    nonPure = true; 
                }
            }
        }
        return true;
    }
};

void DecisionTreeLearner::autoconf(const DataStorage* dataStorage)
{
    setUseBootstrap(true);
    setNumBootstrapExamples(dataStorage->getSize());
    setNumFeatures(std::ceil(std::sqrt(dataStorage->getDimensionality())));
}

/**
 * This class can be used in order to sort the array of data point IDs by
 * a certain dimension
 */
class FeatureComparator {
public:
    /**
     * The feature dimension
     */
    int feature;
    /**
     * The data storage
     */
    DataStorage* storage;
    
    /**
     * Compares two training examples
     */
    bool operator() (const int lhs, const int rhs)
    {
        return storage->getDataPoint(lhs)->at(feature) < storage->getDataPoint(rhs)->at(feature);
    }
};

/**
 * Updates the leaf node histograms using a smoothing parameter
 */
void updateLeafNodeHistogram(std::vector<float> & leafNodeHistograms, const EfficientEntropyHistogram & hist, float smoothing)
{
    leafNodeHistograms.resize(hist.size());
    for (int c = 0; c < hist.size(); c++)
    {
        leafNodeHistograms[c] = std::log((hist.at(c) + smoothing)/(hist.at(c) + hist.size() * smoothing));
    }
}

DecisionTree* DecisionTreeLearner::learn(const DataStorage* dataStorage) const
{
    DataStorage* storage;
    // If we use bootstrap sampling, then this array contains the results of 
    // the sampler. We use it later in order to refine the leaf node histograms
    std::vector<bool> sampled;
    
    if (useBootstrap)
    {
        storage = new DataStorage;
        dataStorage->bootstrap(numBootstrapExamples, storage, sampled);
    }
    else
    {
        storage = new DataStorage(*dataStorage);
    }
    
    // Get the number of training examples and the dimensionality of the data set
    const int D = storage->getDimensionality();
    const int C = storage->getClasscount();
    
    // Set up a new tree. 
    DecisionTree* tree = new DecisionTree();
    
    // This is the list of nodes that still have to be split
    std::vector<int> splitStack;
    splitStack.reserve(static_cast<int>(fastlog2(storage->getSize())));
    
    // Add the root node to the list of nodes that still have to be split
    splitStack.push_back(0);
    
    // This matrix stores the training examples for certain nodes. 
    std::vector< int* > trainingExamples;
    std::vector< int > trainingExamplesSizes;
    trainingExamples.reserve(GRAPH_BUFFER_SIZE);
    trainingExamplesSizes.reserve(GRAPH_BUFFER_SIZE);
    
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
    std::uniform_int_distribution<int> featureDist(0, D - 1);
    
    // We keep track on the depth of each node in this array
    // This allows us to stop splitting after a certain depth is reached
    std::vector<int> depths;
    depths.reserve(GRAPH_BUFFER_SIZE);
    // The root node has depth 0
    depths.push_back(0);
    
    // We use this in order to sort the data points
    FeatureComparator cp;
    cp.storage = storage;
    
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
            hist.add1(storage->getClassLabel(trainingExampleList[m]));
        }

        // Don't split this node
        //  If the number of examples is too small
        //  If the training examples are all of the same class
        //  If the maximum depth is reached
        if (hist.getMass() < minSplitExamples || hist.isPure() || depths[node] > maxDepth)
        {
            delete[] trainingExampleList;
            // Resize and initialize the leaf node histogram
            updateLeafNodeHistogram(tree->getHistogram(node), hist, smoothingParameter);
            continue;
        }
        
        hist.initEntropies();
        
        // These are the parameters we optimize
        float bestThreshold = 0;
        int bestFeature = -1;
        float bestObjective = 1e35;
        int bestLeftMass = 0;
        int bestRightMass = N;

        // Optimize over all features
        for (int f = 0; f < numFeatures; f++)
        {
            const int feature = featureDist(g);
            
            cp.feature = feature;
            std::sort(trainingExampleList, trainingExampleList + N, cp);
            
            // Initialize the histograms
            leftHistogram.reset();
            rightHistogram = hist;
            
            float leftValue = storage->getDataPoint(trainingExampleList[0])->at(feature);
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
                const float rightValue = storage->getDataPoint(n)->at(feature);
                
                // Skip this split, if the two points lie too close together
                const float diff = rightValue - leftValue;
                
                if (diff < 1e-6f)
                {
                    leftValue = rightValue;
                    leftClass = storage->getClassLabel(n);
                    continue;
                }
                
                // Get the objective function
                const float localObjective = leftHistogram.entropy() + rightHistogram.entropy();
                
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
        if (bestFeature < 0|| bestLeftMass < minChildSplitExamples || bestRightMass < minChildSplitExamples)
        {
            // We didn't
            // Don't split
            delete[] trainingExampleList;
            updateLeafNodeHistogram(tree->getHistogram(node), hist, smoothingParameter);
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
            const float featureValue = storage->getDataPoint(n)->at(bestFeature);
            
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
        
        // Update the depth
        depths.push_back(depths[node] + 1);
        depths.push_back(depths[node] + 1);
        
        // Prepare to split the child nodes
        splitStack.push_back(leftChild);
        splitStack.push_back(leftChild + 1);
        
        delete[] trainingExampleList;
    }
    
    learnLogClassPriors(tree, storage);
    
    // Free the data set
    delete storage;
    
    return tree;
}

void DecisionTreeLearner::updateHistograms(DecisionTree* tree, const DataStorage* storage) const
{
    // First: Update the priors
    learnLogClassPriors(tree, storage);
    
    // Now update the histograms
    for (int i = 0; i < storage->getSize(); i++)
    {
        const int leafNode = tree->findLeafNode(storage->getDataPoint(i));
        tree->getHistogram(leafNode)[storage->getClassLabel(i)] += 1;
    }
    
    // Normalize and apply the smoothing
    for (int n = 0; n < tree->getNumNodes(); n++)
    {
        // Only do something if this is a leaf node
        if (!tree->isLeafNode(n))
        {
            continue;
        }
        
        // Ok, this is a leaf node. Get the normalization constant
        float normalization = 0;
        std::vector<float> & hist = tree->getHistogram(n);
        const int C = static_cast<int>(hist.size());
        for (int c = 0; c < C; c++)
        {
            normalization += hist[c];
        }
        // Normalize + log
        for (int c = 0; c < C; c++)
        {
            hist[c] = (hist[c] + smoothingParameter)/(normalization + C*smoothingParameter);
        }
    }
}

void DecisionTreeLearner::dumpSetting(std::ostream & stream) const
{
    stream << std::setw(30) << "Learner" << ": DecisionTreeLearner" << "\n";
    stream << std::setw(30) << "Bootstrap Sampling" << ": " << getUseBootstrap() << "\n";
    stream << std::setw(30) << "Bootstrap Samples" << ": " << getNumBootstrapExamples() << "\n";
    stream << std::setw(30) << "Feature evaluations" << ": " << getNumFeatures() << "\n";
    stream << std::setw(30) << "Max depth" << ": " << getMaxDepth() << "\n";
    stream << std::setw(30) << "Minimum Split Examples" << ": " << getMinSplitExamples() << "\n";
    stream << std::setw(30) << "Minimum Child Split Examples" << ": " << getMinChildSplitExamples() << "\n";
    stream << std::setw(30) << "Smoothing Parameter" << ": " << getSmoothingParameter() << "\n";
}

////////////////////////////////////////////////////////////////////////////////
/// RandomForestLearner
////////////////////////////////////////////////////////////////////////////////

RandomForest* RandomForestLearner::learn(const DataStorage* storage) const
{
    // Set up the empty random forest
    RandomForest* forest = new RandomForest();
    
    // Set up the state for the call backs
    RandomForestLearnerState state;
    state.learner = this;
    state.forest = forest;
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
        
        // Learn the tree
        DecisionTree* tree = treeLearner->learn(storage);
        // Add it to the forest
        #pragma omp critical
        {
            state.tree = ++treeFinishCounter;
            state.action = ACTION_FINISH_TREE;
            evokeCallback(forest, treeFinishCounter - 1, &state);
            forest->addTree(tree);
        }
    }
    
    this->learnLogClassPriors(forest, storage);
    
    state.tree = 0;
    state.action = ACTION_FINISH_FOREST;
    evokeCallback(forest, 0, &state);
    
    return forest;
}

void RandomForestLearner::dumpSetting(std::ostream& stream) const
{
    stream << std::setw(30) << "Learner" << ": RandomForestLearner" << "\n";
    stream << std::setw(30) << "Number of trees" << ": " << getNumTrees() << "\n";
    stream << std::setw(30) << "Number of threads" << ": " << getNumThreads() << "\n";
    stream << "Tree learner settings" << "\n";
    treeLearner->dumpSetting(stream);
}

int RandomForestLearner::defaultCallback(RandomForest* forest, RandomForestLearnerState* state)
{
    switch (state->action) {
        case RandomForestLearner::ACTION_START_FOREST:
            std::cout << "Start random forest training\n";
            state->learner->dumpSetting();
            std::cout << "\n";
            break;
        case RandomForestLearner::ACTION_START_TREE:
            std::cout   << std::setw(15) << std::left << "Start tree " 
                        << std::setw(4) << std::right << state->tree 
                        << " out of " 
                        << std::setw(4) << state->learner->getNumTrees() << "\n";
            break;
        case RandomForestLearner::ACTION_FINISH_TREE:
            std::cout   << std::setw(15) << std::left << "Finish tree " 
                        << std::setw(4) << std::right << state->tree 
                        << " out of " 
                        << std::setw(4) << state->learner->getNumTrees() << "\n";
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