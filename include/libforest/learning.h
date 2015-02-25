#ifndef LIBF_LEARNING_H
#define LIBF_LEARNING_H

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
#include <functional>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>

#include "classifiers.h"

namespace libf {
    /**
     * Forward declarations to reduce compile time
     */
    class DataStorage;
    class DecisionTree;
    class RandomForest;
    class BoostedRandomForest;
    class RandomForestLearner;
    class BoostedRandomForestLearner;
    
    /**
     * AbstractLearner: Combines all common functionality of offline and online
     * learners.
     */
    template<class T, class S>
    class AbstractLearner {
    public:
        /**
         * Registers a callback function that is called every cycle iterations. 
         */
        void addCallback(std::function<int(T*, S*)> callback, int cycle)
        {
            callbacks.push_back(callback);
            callbackCycles.push_back(cycle);
        }
        
        /**
         * The autoconf function should set up the learner such that without
         * any additional settings people can try a learner on a data set. 
         */
        virtual void autoconf(const DataStorage* storage) = 0;
        
        /**
         * Dumps the current settings
         */
        virtual void dumpSetting(std::ostream & stream = std::cout) const = 0;
        
    protected:
        /**
         * Calls the callbacks. The results of the callbacks are bitwise or'ed
         */
        int evokeCallback(T* learnedObject, int iteration, S* state) const
        {
            int result = 0;
            
            // Check all callbacks
            for (size_t i = 0; i < callbacks.size(); i++)
            {
                if ((iteration % callbackCycles[i]) == 0 )
                {
                    // It's time to call this function 
                    result = result | callbacks[i](learnedObject, state);
                }
            }
            
            return result;
        }
        
    private:
        /**
         * The callback functions.
         */
        std::vector<std::function<int(T*, S*)>  > callbacks;
        /**
         * The learning cycle. The callback is called very cycle iterations
         */
        std::vector<int> callbackCycles;
    };
    
    /**
     * This is the base class for all offline learners. It allows you to set a callback
     * function that is called very n iterations of the respective training
     * algorithm.
     */
    template <class T, class S>
    class Learner : public AbstractLearner<T, S> {
    public:
        
        /**
         * Learns a classifier.
         */
        virtual T* learn(const DataStorage* storage) const = 0;

    };
    
    /**
     * This is an ordinary decision tree learning algorithm. It learns the tree
     * using the information gain criterion. In order to make learning easier, 
     * simply use the autoconf option. 
     */
    class DecisionTreeLearner : public Learner<DecisionTree, void> {
    public:
        DecisionTreeLearner() : 
                useBootstrap(true), 
                numBootstrapExamples(10000), 
                numFeatures(10), 
                maxDepth(100), 
                minSplitExamples(3),
                minChildSplitExamples(1), 
                smoothingParameter(1) {}
                
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
        virtual void autoconf(const DataStorage* storage);
        
        /**
         * Learns a decision tree on a data set. If you want to make learning
         * easier, just use the autoconf option before learning. 
         */
        virtual DecisionTree* learn(const DataStorage* storage) const;
        
        /**
         * Dumps the settings
         */
        virtual void dumpSetting(std::ostream & stream = std::cout) const;
        
        /**
         * Sets the smoothing parameter
         */
        void setSmoothingParameter(float _smoothingParameter)
        {
            smoothingParameter = _smoothingParameter;
        }
        
        /**
         * Returns the smoothing parameter
         */
        float getSmoothingParameter() const
        {
            return smoothingParameter;
        }
        
        /**
         * Updates the histograms
         */
        void updateHistograms(DecisionTree* tree, const DataStorage* storage) const;
        
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
        /**
         * The smoothing parameter for the histograms
         */
        float smoothingParameter;
    };
    
    /**
     * This class holds the current state of the random forest learning
     * algorithm.
     */
    class RandomForestLearnerState {
    public:
        RandomForestLearnerState() : action(0), learner(0), forest(0), tree(0), startTime(std::chrono::high_resolution_clock::now()) {}
        
        /**
         * The current action
         */
        int action;
        /**
         * The learner object
         */
        const RandomForestLearner* learner;
        /**
         * The learned object
         */
        const RandomForest* forest;
        /**
         * The current tree
         */
        int tree;
        /**
         * The start time
         */
        std::chrono::high_resolution_clock::time_point startTime;
        
        /**
         * Returns the passed time in microseconds
         */
        std::chrono::microseconds getPassedTime()
        {
            std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>( now - startTime );
        }
    };
    
    /**
     * This is a random forest learner. 
     */
    class RandomForestLearner : public Learner<RandomForest, RandomForestLearnerState> {
    public:
        /**
         * These are the actions of the learning algorithm that are passed
         * to the callback functions.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_FINISH_TREE = 2;
        const static int ACTION_START_FOREST = 3;
        const static int ACTION_FINISH_FOREST = 4;
        
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(RandomForest* forest, RandomForestLearnerState* state);
        
        RandomForestLearner() : numTrees(8), treeLearner(0), numThreads(1) {}
        
        /**
         * Sets the number of trees. 
         */
        void setNumTrees(int _numTrees)
        {
            assert(_numTrees >= 1);
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
        virtual RandomForest* learn(const DataStorage* storage) const;
        
        /**
         * The autoconf function should set up the learner such that without
         * any additional settings people can try a learner on a data set. 
         */
        virtual void autoconf(const DataStorage* storage) {}
        
        /**
         * Dumps the settings
         */
        virtual void dumpSetting(std::ostream & stream = std::cout) const;
        
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
     * This class holds the current state of the boosted random forest learning
     * algorithm.
     */
    class BoostedRandomForestLearnerState {
    public:
        BoostedRandomForestLearnerState() : action(0), learner(0), forest(0), tree(0), error(0), alpha(0), startTime(std::chrono::high_resolution_clock::now()) {}
        
        /**
         * The current action
         */
        int action;
        /**
         * The learner object
         */
        const BoostedRandomForestLearner* learner;
        /**
         * The learned object
         */
        const BoostedRandomForest* forest;
        /**
         * The current tree
         */
        int tree;
        /**
         * The error value
         */
        float error;
        /**
         * The tree weight
         */
        float alpha;
        /**
         * The start time
         */
        std::chrono::high_resolution_clock::time_point startTime;
        
        /**
         * Returns the passed time in microseconds
         */
        std::chrono::microseconds getPassedTime()
        {
            std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>( now - startTime );
        }
    };
    
    /**
     * This is a random forest learner. 
     */
    class BoostedRandomForestLearner : public Learner<BoostedRandomForest, BoostedRandomForestLearnerState> {
    public:
        /**
         * These are the actions of the learning algorithm that are passed
         * to the callback functions.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_FINISH_TREE = 2;
        const static int ACTION_START_FOREST = 3;
        const static int ACTION_FINISH_FOREST = 4;
        
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(BoostedRandomForest* forest, BoostedRandomForestLearnerState* state);
        
        BoostedRandomForestLearner() : numTrees(8), treeLearner(0) {}
        
        /**
         * Sets the number of trees. 
         */
        void setNumTrees(int _numTrees)
        {
            assert(_numTrees >= 1);
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
         * Learns a forests. 
         */
        virtual BoostedRandomForest* learn(const DataStorage* storage) const;
        
        /**
         * The autoconf function should set up the learner such that without
         * any additional settings people can try a learner on a data set. 
         */
        virtual void autoconf(const DataStorage* storage) {}
        
        /**
         * Dumps the settings
         */
        virtual void dumpSetting(std::ostream & stream = std::cout) const;
        
    private:
        /**
         * The number of trees that we shall learn
         */
        int numTrees;
        /**
         * The tree learner
         */
        DecisionTreeLearner* treeLearner;
    };
}

#endif
