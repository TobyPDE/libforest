#include "lib_mcmcf/learning.h"
#include "lib_mcmcf/data.h"
#include "lib_mcmcf/classifiers.h"
#include "fastlog.h"
#include "mcmc.h"

#include <algorithm>
#include <random>
#include <map>
#include <ctime>

using namespace mcmcf;

#define ENTROPY(p) (-(p)*fastlog2(p))

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
    int bins;

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

                entropies[i] = ENTROPY(at(i));

                totalEntropy += entropies[i];
            }
        }
    }

    /**
     * Sets all entries in the histogram to 0
     */
    void reset()
    {
        // Only reset the histogram if there are more than 0 bins
        if (histogram == 0) return;

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
        int maxCount = at(0);
        for (int i = 1; i < bins; i++)
        {
            if (at(i) > maxCount)
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
            if (at(i) > 0)
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
    setUseRandomFeatures(true);
    setNumBootstrapExamples(dataStorage->getSize());
    setNumFeatures(std::ceil(std::sqrt(dataStorage->getDimensionality())));
}

DecisionTree* DecisionTreeLearner::learn(const DataStorage* dataStorage) const
{
    DataStorage* storage;
    
    if (useBootstrap)
    {
        storage = new DataStorage;
        dataStorage->bootstrap(numBootstrapExamples, storage);
    }
    else
    {
        storage = new DataStorage(*dataStorage);
    }
    storage->computeIntClassLabels(dataStorage);
    
    // Get the number of training examples and the dimensionality of the data set
    const int D = storage->getDimensionality();
    const int C = storage->getClasscount();
    
    // Set up a new tree. 
    DecisionTree* tree = new DecisionTree();
    
    // This is the list of nodes that still have to be split
    std::vector<int> splitQueue;
    
    // Add the root node to the list of nodes that still have to be split
    splitQueue.push_back(0);
    
    // This matrix stores the training examples for certain nodes. 
    std::vector< std::vector<int> > trainingExamples;
    
    // Add all training example to the root node
    trainingExamples.push_back(std::vector<int>(storage->getSize()));
    for (int n = 0; n < storage->getSize(); n++)
    {
        trainingExamples[0][n] = n;
    }
    
    // We use these arrays during training for the left and right histograms
    EfficientEntropyHistogram leftHistogram(C);
    EfficientEntropyHistogram rightHistogram(C);
    
    // Set up a probability distribution over the features
    std::mt19937 g(time(0));
    std::uniform_int_distribution<int> featureDist(0, D - 1);
    
    // We keep track on the depth of each node in this array
    // This allows us to stop splitting after a certain depth is reached
    std::vector<int> depths;
    // The root node has depth 0
    depths.push_back(0);
    
    // Start training
    while (splitQueue.size() > 0)
    {
        // Extract an element from the queue
        const int node = splitQueue.back();
        splitQueue.pop_back();
        
        // Get the training example list
        std::vector<int> & trainingExampleList = trainingExamples[node];
        const int N = static_cast<int>(trainingExampleList.size());

        // Set up the right histogram
        // Because we start with the threshold being at the left most position
        // The right child node contains all training examples
        
        EfficientEntropyHistogram hist(C);
        for (int m = 0; m < N; m++)
        {
            // Get the class label of this training example
            const int n = trainingExampleList[m];
            hist.add(storage->getIntClassLabel(n), 1);
        }

        // Determine the class label for this node
        tree->setClassLabel(node, hist.argMax());
        
        // Don't split this node
        //  If the number of examples is too small
        //  If the training examples are all of the same class
        //  If the maximum depth is reacherd
        if (hist.getMass() < minSplitExamples || hist.isPure() || depths[node] > maxDepth)
        {
            trainingExampleList.clear();
            continue;
        }
        
        hist.initEntropies();
        
        // These are the parameters we optimize
        float bestThreshold = 0;
        int bestFeature = 0;
        float bestObjective = 1e35;
        int bestLeftMass = 0;
        int bestRightMass = N;

        // Optimize over all features
        for (int f = 0; f < numFeatures; f++)
        {
            int feature = f;
            // Sample a random feature if this is required
            if (useRandomFeatures)
            {
                feature = featureDist(g);
            }
            std::sort(trainingExampleList.begin(), trainingExampleList.end(), [storage, feature](const int & lhs, const int & rhs) -> bool {
                return storage->getDataPoint(lhs)->at(feature) < storage->getDataPoint(rhs)->at(feature);
            });
            
            // Initialize the histograms
            leftHistogram.reset();
            rightHistogram = hist;
            
            float leftValue = storage->getDataPoint(trainingExampleList[0])->at(feature);
            int leftClass = storage->getIntClassLabel(trainingExampleList[0]);
            
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
                    leftClass = storage->getIntClassLabel(n);
                    continue;
                }
                
                // Get the objective function
                const float localObjective = leftHistogram.entropy() + rightHistogram.entropy();
                
                if (localObjective < bestObjective)
                {
                    // Get the threshold value
                    bestThreshold = 0.5f * (leftValue + rightValue);
                    bestFeature = feature;
                    bestObjective = localObjective;
                    bestLeftMass = leftHistogram.getMass();
                    bestRightMass = rightHistogram.getMass();
                }
                
                leftValue = rightValue;
                leftClass = storage->getIntClassLabel(n);
            }
        }
        
        // Did we find good split values?
        if (bestObjective > 1e34 || bestLeftMass < minChildSplitExamples || bestRightMass < minChildSplitExamples)
        {
            // We didn't
            // Don't split
            trainingExampleList.clear();
            continue;
        }
        
        // Set up the data lists for the child nodes
        trainingExamples.push_back(std::vector<int>(bestLeftMass));
        trainingExamples.push_back(std::vector<int>(bestRightMass));
        
        std::vector<int> & list = trainingExamples[node];
        std::vector<int> & leftList = trainingExamples[trainingExamples.size() - 2];
        std::vector<int> & rightList = trainingExamples[trainingExamples.size() - 1];
        
        // Sort the points
        for (int m = 0; m < N; m++)
        {
            const int n = list[m];
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
        splitQueue.push_back(leftChild);
        splitQueue.push_back(leftChild + 1);
        
        trainingExamples[node].clear();
    }
    
    // Free the data set
    delete storage;
    
    return tree;
}

////////////////////////////////////////////////////////////////////////////////
/// RandomForestLearner
////////////////////////////////////////////////////////////////////////////////

RandomForest* RandomForestLearner::learn(const DataStorage* storage) const
{
    // Set up the empty random forest
    RandomForest* forest = new RandomForest();
    
    #pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < numTrees; i++)
    {
        std::cout << i << "\n";
        // Learn the tree
        DecisionTree* tree = treeLearner->learn(storage);
        // Add it to the forest
        #pragma omp critical
        {
            forest->addTree(tree);
        }
    }
    return forest;
}


////////////////////////////////////////////////////////////////////////////////
/// RandomForestPrune
////////////////////////////////////////////////////////////////////////////////

/**
 * This is the type of the state vector for the pruning
 */
typedef std::vector<int> StateType;

/**
 * This is the simulated annealing callback. It just logs everything to the 
 * console
 */
class PruneCallback : public SACallback<StateType> {
public:
    /**
     * The function that is called
     */
    virtual int callback(const StateType & state, float energy, int iteration, float temperature)
    {
        std::cout << "iteration: " << iteration << " energy: " << energy << " temp: " << temperature << std::endl;
        return 0;
    }
};

/**
 * This is the move that reduces the dimensionality of the state by removing
 * a random tree from the ensemble.
 */
class PruneMoveRemove : public SAMove<StateType> {
public:
    PruneMoveRemove() : g(time(0)) {}
    
    /**
     * Computes the move
     */
    void move(const StateType & state, StateType & newState)
    {
        // Only remove something if the state has at least two trees
        if (state.size() < 2)
        {
            // Just copy the state
            newState = state;
        }
        else
        {
            // Set up a probability distribution
            std::uniform_int_distribution<int> dist(0, static_cast<int>(state.size() - 1));
            // Decide which tree to remove
            const int removeTree = dist(g);
            // Copy all other trees to the new state
            for (int i = 0; i < static_cast<int>(state.size()); i++)
            {
                if (i == removeTree)
                {
                    // Don't add the tree that is supposed to be removed
                    continue;
                }
                else
                {
                    // Add the tree
                    newState.push_back(state[i]);
                }
            }
        }
    }
private:
    /**
     * This is the random number generator
     */
    std::mt19937 g;
};

/**
 * Adds a random tree to the ensemble. 
 */

/**
 * This is the move that reduces the dimensionality of the state by removing
 * a random tree from the ensemble.
 */
class PruneMoveAdd : public SAMove<StateType> {
public:
    PruneMoveAdd(RandomForest* forest) : g(time(0)), dist(0, forest->getSize() - 1) {}
    
    /**
     * Computes the move
     */
    void move(const StateType & state, StateType & newState)
    {
        // Copy all other trees
        newState = state;
        
        // Add a random tree
        const int tree = dist(g);
        newState.push_back(tree);
    }
    
private:
    /**
     * This is the random number generator
     */
    std::mt19937 g;
    /**
     * This is a distribution over the trees
     */
    std::uniform_int_distribution<int> dist;
};

/**
 * Exchanges a tree with some other tree
 */
class PruneMoveExchange : public SAMove<StateType> {
public:
    PruneMoveExchange(RandomForest* forest) : g(time(0)), dist(0, forest->getSize() - 1) {}
    
    /**
     * Computes the move
     */
    void move(const StateType & state, StateType & newState)
    {
        // Copy all other trees
        newState = state;
        
        // Set up a distribution over the current state
        std::uniform_int_distribution<int> stateDist(0, static_cast<int>(state.size() - 1));
        
        // Choose a tree to add
        const int addTree = dist(g);
        // Choose a tree to replace
        const int replaceTree = stateDist(g);
        
        newState[replaceTree] = addTree;
    }
    
private:
    /**
     * This is the random number generator
     */
    std::mt19937 g;
    /**
     * This is a distribution over the trees
     */
    std::uniform_int_distribution<int> dist;
};

/**
 * This is the energy function for the pruning optimization
 */
class PruneEnergy : public SAEnergy<StateType> {
public:
    PruneEnergy(RandomForest* forest, DataStorage* storage) : storage(storage) {
        // Determine the predictions
        for (int i = 0; i < forest->getSize(); i++)
        {
            DecisionTree* tree = forest->getTree(i);
            predictions.push_back(std::vector<int>());
            tree->classify(storage, predictions[i]);
        }
    }
    
    /**
     * Computes the energy
     */
    virtual float energy(const StateType & state)
    {
        // Calculate the prediction for the current ensemble and every data point
        float error = 0;
        for (int n = 0; n < storage->getSize(); n++)
        {
            // Get the votes from each tree
            std::vector<int> votes(storage->getClasscount());
            // Initialize the votes
            for (int c = 0; c < storage->getClasscount(); c++)
            {
                votes[c] = 0;
            }
            
            // Keep track on the maximum
            int maxLabel = 0;
            int maxVotes = 0;
            // Cast the votes
            for (size_t j = 0; j < state.size(); j++)
            {
                const int vote = predictions[j][n];
                votes[vote]++;
                
                if (votes[vote] > maxVotes)
                {
                    maxVotes = votes[vote];
                    maxLabel = vote;
                }
            }
            
            // Was the label predicted correctly?
            if (maxLabel != storage->getIntClassLabel(n))
            {
                // Nope, it wasn't
                error += 1;
            }
        }
        
        // Return the normalized error
        return error/storage->getSize();
    }
    
private:
    /**
     * These are the predictions for every data point and every tree
     */
    std::vector< std::vector<int> > predictions;
    /**
     * This is the storage
     */
    DataStorage* storage;
};

void RandomForestPrune::prune(RandomForest* forest, DataStorage* storage) const
{
    // We prune the forest using simulated annealing
    SimulatedAnnealing<std::vector<int> > sa;
    
    // Set up the cooling schedule
    GeometricCoolingSchedule schedule;
    schedule.setAlpha(0.9f);
    sa.setCoolingSchedule(&schedule);
    
    // Set up the moves
    PruneMoveAdd addMove(forest);
    PruneMoveRemove removeMove;
    PruneMoveExchange exchangeMove(forest);
    sa.addMove(&addMove, 0.02f);
    sa.addMove(&removeMove, 0.02f);
    sa.addMove(&exchangeMove, 0.96f);
    
    // Set up the energy function
    PruneEnergy energy(forest, storage);
    sa.setEnergyFunction(&energy);
    
    // Set up the callback function
    PruneCallback callback;
    sa.addCallback(&callback);
    
    // Find some initial state
    StateType state;
    state.push_back(0);
    
    // Optimize the stuff
    sa.optimize(state);
}