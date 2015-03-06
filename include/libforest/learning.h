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
    class DecisionTreeLearner;
    class RandomForestLearner;
    class BoostedRandomForestLearner;
    
    /**
     * AbstractLearner: Combines all common functionality of offline and online
     * learners. It allows you to set a callback function that is called very n
     * iterations of the respective training algorithm.
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
     * Abstract learner state for measuring time and defining actions.
     */
    class AbstractLearnerState {
    public:
        AbstractLearnerState() : 
                action(0),
                startTime(std::chrono::high_resolution_clock::now()) {}
        
        /**
         * The current action
         */
        int action;
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
     * This is the base class for all offline learners.
     */
    template<class T>
    class Learner {
    public:
        
        /**
         * Learns a classifier.
         */
        virtual T* learn(const DataStorage* storage) = 0;
    };
    
    /**
     * This is an abstract decision tree learning providing functionality
     * needed for all decision tree learners (online or offline.
     */
    template<class S>
    class AbstractDecisionTreeLearner : public AbstractLearner<DecisionTree, S> {
    public:
        AbstractDecisionTreeLearner() : 
                numFeatures(10), 
                maxDepth(100), 
                minSplitExamples(3),
                minChildSplitExamples(1), 
                smoothingParameter(1),
                useBootstrap(false) {}
                
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
         * Get the Mean Decrease Impurity importance.
         * 
         * @see http://orbi.ulg.ac.be/bitstream/2268/170309/1/thesis.pdf
         */
        float getMDIImportance(int feature) const
        {
            return impurityDecrease[feature];
        }
        
        /**
         * Get the Mean Impurity Decrease variable importance for all features, 
         * see above.
         */
        std::vector<float> & getMDIImportance()
        {
            return impurityDecrease;
        }
        
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
        
    protected:
        
        /**
         * Splits a node
         */
        void splitNode(DecisionTree* tree, int node) const;
        
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
        /**
         * Whether or not bootstrapping shall be used
         */
        bool useBootstrap;
        /**
         * The sum of impurity decrease per feature
         */
        std::vector<float> impurityDecrease;
    };
    
    class DecisionTreeLearnerState : public AbstractLearnerState {
    public:
        DecisionTreeLearnerState() : AbstractLearnerState(),
                objective(0), 
                depth(0) {}
        
        /**
         * Objective of splitted node.
         */
        float objective;
        /**
         * Depth of spitted node.
         */
        int depth;
    };
    
    /**
     * This is an ordinary offline decision tree learning algorithm. It learns the
     * tree using the information gain criterion.
     */
    class DecisionTreeLearner : public AbstractDecisionTreeLearner<DecisionTreeLearnerState>, 
            public Learner<DecisionTree> {
    public:
        DecisionTreeLearner() : AbstractDecisionTreeLearner(),
                numBootstrapExamples(1) {}
                
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(DecisionTree* tree, DecisionTreeLearnerState* state);
                
        /**
         * Actions for the callback function.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_SPLIT_NODE = 2;
        
        /**
         * Sets the number of samples to use for bootstrapping.
         */
        void setNumBootstrapExamples(int _numBootstrapExamples)
        {
            numBootstrapExamples = _numBootstrapExamples;
        }
        
        /**
         * Returns the number of samples used for bootstrapping.
         */
        int getNumBootstrapExamples() const
        {
            return numBootstrapExamples;
        }
        
        /**
         * Learns a decision tree on a data set.
         */
        virtual DecisionTree* learn(const DataStorage* storage);
        
        /**
         * Updates the histograms
         */
        void updateHistograms(DecisionTree* tree, const DataStorage* storage) const;
        
    protected:
        
        /**
         * The number of bootstrap examples that shall be used.
         */
        int numBootstrapExamples;
    };
    
    /**
     * This is a an abstract random forest learner providing functionality for
     * online and offline learning.
     */
    template<class S>
    class AbstractRandomForestLearner : public AbstractLearner<RandomForest, S> {
    public:
        
        AbstractRandomForestLearner() : numTrees(8), numThreads(1) {}
        
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
         * Get the Mean Impurity Decrease variable importance.
         * 
         * @see http://orbi.ulg.ac.be/bitstream/2268/170309/1/thesis.pdf
         */
        float getMDIImportance(int feature) const
        {
            return impurityDecrease[feature];
        }
        
        /**
         * Get the Mean Impurity Decrease variable importance for all features, 
         * see above.
         */
        std::vector<float> & getMDIImportance()
        {
            return impurityDecrease;
        }
        
    protected:
        /**
         * The number of trees that we shall learn
         */
        int numTrees;
        /**
         * The number of threads that shall be used to learn the forest
         */
        int numThreads;
        /**
         * Used to compute the Mean Decrease Importance variable importance
         * if requested through the tree learner.
         */
        std::vector<float> impurityDecrease;
    };
    
    /**
     * This class holds the current state of the random forest learning
     * algorithm.
     */
    class RandomForestLearnerState : public AbstractLearnerState {
    public:
        RandomForestLearnerState() : AbstractLearnerState(),
                tree(0),
                numTrees(0) {}
        
        /**
         * The current tree
         */
        int tree;
        /**
         * Total number of trees learned.
         */
        int numTrees;
    };
    
    /**
     * This is an offline random forest learner. 
     */
    class RandomForestLearner : public AbstractRandomForestLearner<RandomForestLearnerState>,
            public Learner<RandomForest> {
    public:
        
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(RandomForest* forest, RandomForestLearnerState* state);
        
        /**
         * These are the actions of the learning algorithm that are passed
         * to the callback functions.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_FINISH_TREE = 2;
        const static int ACTION_START_FOREST = 3;
        const static int ACTION_FINISH_FOREST = 4;
        
        RandomForestLearner() : AbstractRandomForestLearner(),
                treeLearner(0) {}
        
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
        virtual RandomForest* learn(const DataStorage* storage);

    protected:
        /**
         * The tree learner
         */
        DecisionTreeLearner* treeLearner;
    };
    
    /**
     * This is a random forest learner. 
     */
    template<class S>
    class AbstractBoostedRandomForestLearner : public AbstractLearner<BoostedRandomForest, S> {
    public:
        
        AbstractBoostedRandomForestLearner() : numTrees(8) {}
        
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
        
    protected:
        /**
         * The number of trees that we shall learn
         */
        int numTrees;
        /**
         * The tree learner
         */
        DecisionTreeLearner* treeLearner;
    };
    
    /**
     * This class holds the current state of the boosted random forest learning
     * algorithm.
     */
    class BoostedRandomForestLearnerState : public AbstractLearnerState {
    public:
        BoostedRandomForestLearnerState() : AbstractLearnerState(),
                tree(0),
                numTrees(0),
                error(0), 
                alpha(0) {}
        
        /**
         * The current tree
         */
        int tree;
        /**
         * Number of learned trees.
         */
        int numTrees;
        /**
         * The error value
         */
        float error;
        /**
         * The tree weight
         */
        float alpha;
    };
    
    /**
     * This is a random forest learner. 
     */
    class BoostedRandomForestLearner :
            public AbstractBoostedRandomForestLearner<BoostedRandomForestLearnerState>,
            public Learner<BoostedRandomForest> {
    public:
        
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(BoostedRandomForest* forest, BoostedRandomForestLearnerState* state);
        
        /**
         * These are the actions of the learning algorithm that are passed
         * to the callback functions.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_FINISH_TREE = 2;
        const static int ACTION_START_FOREST = 3;
        const static int ACTION_FINISH_FOREST = 4;
        
        BoostedRandomForestLearner() : AbstractBoostedRandomForestLearner(),
                treeLearner(0) {}
        
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
        virtual BoostedRandomForest* learn(const DataStorage* storage);
        
    protected:
        /**
         * The tree learner
         */
        DecisionTreeLearner* treeLearner;
    };
}

#endif
