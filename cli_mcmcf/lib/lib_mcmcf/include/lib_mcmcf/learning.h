#ifndef MCMCF_LEARNING_H
#define MCMCF_LEARNING_H

/**
 * This file implements the actual learning algorithms. There are basically 
 * four major algorithm:
 * 1. Ordinary decision tree learning
 * 2. Ordinary random forest learning
 * 3. Decorellated MCMC based decision tree learning
 * 4. Decorrelated MCMC based random forest learning
 * The MCMC based algorithm depend on the Metropolis-Hastings algorithm as well
 * as its variant Simulated Annealing. Both of these algorithms can be found in 
 * mcmc.h. 
 * All learning algorithm use a special data structure to speed up the process. 
 * The data structure sorts the data set according to each dimension.
 */
#include <cassert>

#include "data.h"
#include "classifiers.h"

namespace mcmcf {
    /**
     * Forward declarations to reduce compile time
     */
    class DataStorage;
    class DecisionTree;
    class RandomForest;

    /**
     * This is the base class for all learners. It allows you to set a callback
     * function that is called very n iterations of the respective training
     * algorithm.
     */
    template <class T>
    class Learner {
    public:
        /**
         * Registers a callback function that is called every cycle iterations. 
         */
        void addCallback(int (*callback)(T* learnedObject, int iteration), int cycle)
        {
            callbacks.push_back(callback);
            callbackCycles.push_back(cycle);
        }
        
    protected:
        /**
         * Calls the callbacks. The results of the callbacks are bitwise or'ed
         */
        int evokeCallback(T* learnedObject, int iteration) const
        {
            int result = 0;
            
            // Check all callbacks
            for (size_t i = 0; i < callbacks.size(); i++)
            {
                if ((iteration % callbackCycles[i]) == 0 )
                {
                    // It's time to call this function 
                    result = result | callbacks[i](learnedObject, iteration);
                }
            }
            
            return result;
        }
        
    private:
        /**
         * The callback functions.
         */
        std::vector<int (*)(T* learnedObject, int iteration)> callbacks;
        /**
         * The learning cycle. The callback is called very cycle iterations
         */
        std::vector<int> callbackCycles;
    };
    
    /**
     * This is an ordinary decision tree learning algorithm. It learns the tree
     * using the information gain criterion. In order to make learning easier, 
     * simply use the autoconf option. 
     */
    class DecisionTreeLearner : public Learner<DecisionTree> {
    public:
        DecisionTreeLearner() : 
                useBootstrap(true), 
                numBootstrapExamples(10000), 
                useRandomFeatures(true), 
                numFeatures(10), 
                maxDepth(100), 
                minSplitExamples(3),
                minChildSplitExamples(1) {}
                
        /**
         * Sets whether or not bootstrapping shall be used
         */
        void setUseBootstrap(bool _useBootstrap)
        {
            useBootstrap = _useBootstrap;
        }

        /**
         * Returns whether or not bootstrapping is used
         */
        bool getUseBootstrap() const
        {
            return useBootstrap;
        }

        /**
         * Sets the number of features that are required to perform a split
         */
        void setMinSplitExamples(int minSplitExamples) 
        {
            assert(minSplitExamples >= 0);
            this->minSplitExamples = minSplitExamples;
        }

        /**
         * Returns the number of features that are required to perform a split
         */
        int getMinSplitExamples() const 
        {
            return minSplitExamples;
        }

        /**
         * Sets the maximum depth of a tree
         */
        void setMaxDepth(int maxDepth) 
        {
            assert(maxDepth >= 1);
            this->maxDepth = maxDepth;
        }

        /**
         * Returns the maximum depth of a tree
         */
        int getMaxDepth() const 
        {
            return maxDepth;
        }

        /**
         * Sets the number of random features that shall be evaluated
         */
        void setNumFeatures(int numFeatures) 
        {
            assert(numFeatures >= 1);
            this->numFeatures = numFeatures;
        }

        /**
         * Returns the number of random features that shall be evaluated
         */
        int getNumFeatures() const 
        {
            return numFeatures;
        }

        /**
         * Sets whether or not random features shall be used.
         */
        void setUseRandomFeatures(bool useRandomFeatures) 
        {
            this->useRandomFeatures = useRandomFeatures;
        }

        /**
         * Returns whether or not random features shall be used.
         */
        bool getUseRandomFeatures() const 
        {
            return useRandomFeatures;
        }

        /**
         * Sets the number of bootstrap examples.
         */
        void setNumBootstrapExamples(int numBootstrapExamples) 
        {
            assert(numBootstrapExamples >= 1);
            this->numBootstrapExamples = numBootstrapExamples;
        }

        /**
         * Returns the number of bootstrap examples.
         */
        int getNumBootstrapExamples() const 
        {
            return numBootstrapExamples;
        }
        
        /**
         * Sets the minimum number of examples that need to be in both child nodes
         * in order to perform a split. 
         */
        void setMinChildSplitExamples(int _minChildSplitExamples)
        {
            assert(_minChildSplitExamples >= 0);
            minChildSplitExamples = _minChildSplitExamples;
        }
        
        /**
         * Returns the minimum number of examples that need to be in both child nodes
         * in order to perform a split. 
         */
        int getMinChildSplitExamples() const
        {
            return minChildSplitExamples;
        }
        
        /**
         * Configures the learning automatically depending on a certain data
         * set. 
         */
        void autoconf(const DataStorage* storage);
        
        /**
         * Learns a decision tree on a data set. If you want to make learning
         * easier, just use the autoconf option before learning. 
         */
        DecisionTree* learn(const DataStorage* storage) const;
        
    private:
        
        /**
         * Splits a node
         */
        void splitNode(DecisionTree* tree, int node) const;
        
        /**
         * Whether or not bootstrapping shall be used
         */
        bool useBootstrap;
        /**
         * The number of bootstrap examples that shall be used.
         */
        int numBootstrapExamples;
        /**
         * Whether or not only a random selection of features shall be 
         * evaluated for each split.
         */
        bool useRandomFeatures;
        /**
         * The number of random features that shall be evaluated for each 
         * split.
         */
        int numFeatures;
        /**
         * Maximum depth of the tree
         */
        int maxDepth;
        /**
         * The minimum number of training examples in a node that are required
         * in order to split the node further
         */
        int minSplitExamples;
        /**
         * The minimum number of examples that need to be in both child nodes
         * in order to perform a split. 
         */
        int minChildSplitExamples;
    };
    
    /**
     * This is a random forest learner. 
     */
    class RandomForestLearner : public Learner<RandomForest> {
    public:
        RandomForestLearner() : numTrees(8), treeLearner(0), numThreads(1) {}
        
        /**
         * Sets the number of trees. 
         */
        void setNumTrees(int _numTrees)
        {
            assert(_numTrees > 1);
            numTrees = _numTrees;
        }
        
        /**
         * Returns the number of trees
         */
        int getNumTrees() const
        {
            return numTrees;
        }
        
        /**
         * Sets the decision tree learner
         */
        void setTreeLearner(DecisionTreeLearner* _treeLearner)
        {
            treeLearner = _treeLearner;
        }
        
        /**
         * Returns the decision tree learner
         */
        DecisionTreeLearner* getTreeLearner() const
        {
            return treeLearner;
        }
        
        /**
         * Sets the number of threads
         */
        void setNumThreads(int _numThreads)
        {
            numThreads = _numThreads;
        }
        
        /**
         * Returns the number of threads
         */
        int getNumThreads() const
        {
            return numThreads;
        }
        
        /**
         * Learns a forests. 
         */
        RandomForest* learn(const DataStorage* storage) const;
        
    private:
        /**
         * The number of trees that we shall learn
         */
        int numTrees;
        /**
         * The tree learner
         */
        DecisionTreeLearner* treeLearner;
        /**
         * The number of threads that shall be used to learn the forest
         */
        int numThreads;
    };
    
    /**
     * Global random forest pruning. We remove trees from the forest in order to
     * get an optimal error rate. 
     */
    class RandomForestPrune : public Learner<RandomForest> {
    public:
        /**
         * Prunes a random forest using the given data set.
         */
        RandomForest* prune(RandomForest* forest, DataStorage* storage) const;
    };
}

#endif
