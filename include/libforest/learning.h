#ifndef LIBF_LEARNING_H
#define LIBF_LEARNING_H

#include <functional>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>

#include "error_handling.h"
#include "data.h"
#include "classifiers.h"

namespace libf {
    /**
     * Forward declarations to reduce compile time
     */
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
         * 
         * @param callback The callback function
         * @param cycle The number of cycles in between the function calls
         */
        void addCallback(const std::function<int(std::shared_ptr<T>, const S &)> & callback, int cycle)
        {
            callbacks.push_back(callback);
            callbackCycles.push_back(cycle);
        }
        
    protected:
        /**
         * Calls the callbacks. The results of the callbacks are bitwise or'ed
         */
        int evokeCallback(std::shared_ptr<T> learnedObject, int iteration, const S & state) const
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
        std::vector<std::function<int(std::shared_ptr<T>, const S &)>  > callbacks;
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
         * 
         * @return The time passed since instantiating the state object
         */
        std::chrono::microseconds getPassedTime() const
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
         * 
         * @param storage The storage to train the classifier on
         * @return The trained classifier
         */
        virtual std::shared_ptr<T> learn(AbstractDataStorage::ptr storage) = 0;
    };
    
    /**
     * This is an abstract decision tree learning providing functionality
     * needed for all decision tree learners (online or offline.
     */
    template<class M, class S>
    class AbstractDecisionTreeLearner : public AbstractLearner<M, S> {
    public:
        AbstractDecisionTreeLearner() : 
                numFeatures(10), 
                maxDepth(100), 
                minSplitExamples(3),
                minChildSplitExamples(1) {}
                
        /**
         * Sets the number of features that are required to perform a split. If 
         * there are less than the specified number of training examples at a 
         * node, it won't be split and becomes a leaf node. 
         * 
         * @param minSplitExamples The minimum number of examples required to split a node
         */
        void setMinSplitExamples(int minSplitExamples) 
        {
            BOOST_ASSERT(minSplitExamples >= 0);
            this->minSplitExamples = minSplitExamples;
        }

        /**
         * Returns the minimum number of training examples required in order
         * to split a node. 
         * 
         * @return The minimum number of training examples required to split a node
         */
        int getMinSplitExamples() const 
        {
            return minSplitExamples;
        }

        /**
         * Sets the maximum depth of a tree where the root node receives depth
         * 0. 
         * 
         * @param maxDepth the max depth
         */
        void setMaxDepth(int maxDepth) 
        {
            BOOST_ASSERT(maxDepth >= 0);
            
            this->maxDepth = maxDepth;
        }

        /**
         * Returns the maximum depth of a tree where the root node has depth 0. 
         * 
         * @return The maximum depth of a tree
         */
        int getMaxDepth() const 
        {
            return maxDepth;
        }

        /**
         * Sets the number of random features that shall be evaluated. If the
         * number of features equals the total number of dimensions, then we
         * train an ordinary decision tree. 
         * 
         * @param numFeatures The number of features to evaluate. 
         */
        void setNumFeatures(int numFeatures) 
        {
            BOOST_ASSERT(numFeatures >= 1);
            
            this->numFeatures = numFeatures;
        }

        /**
         * Returns the number of random features that shall be evaluated
         * 
         * @return The number of features to evaluate
         */
        int getNumFeatures() const 
        {
            return numFeatures;
        }
        
        /**
         * Sets the minimum number of examples that need to be in both child nodes
         * in order to perform a split. 
         * 
         * @param _minChildSplitExamples The required number of examples
         */
        void setMinChildSplitExamples(int _minChildSplitExamples)
        {
            BOOST_ASSERT(_minChildSplitExamples >= 0);
            minChildSplitExamples = _minChildSplitExamples;
        }
        
        /**
         * Returns the minimum number of examples that need to be in both child nodes
         * in order to perform a split. 
         * 
         * @return The required number of examples
         */
        int getMinChildSplitExamples() const
        {
            return minChildSplitExamples;
        }
        
        /**
         * Get the Mean Decrease Impurity importance.
         * 
         * @see http://orbi.ulg.ac.be/bitstream/2268/170309/1/thesis.pdf
         * @param feature TODO
         * @return TODO
         */
        float getImportance(int feature) const
        {
            BOOST_ASSERT(0 <= feature && feature < importance.size());
            return importance[feature];
        }
        
        /**
         * Get the Mean Impurity Decrease variable importance for all features, 
         * see above.
         * 
         * @return TODO
         */
        std::vector<float> & getImportance()
        {
            return importance;
        }
        
    protected:
        
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
         * The sum of impurity decrease per feature
         */
        std::vector<float> importance;
    };
    
    class DecisionTreeLearnerState : public AbstractLearnerState {
    public:
        DecisionTreeLearnerState() : AbstractLearnerState(),
                objective(0), 
                depth(0) {}
        
        /**
         * Objective of split node.
         */
        float objective;
        /**
         * Depth of spitted node.
         */
        int depth;
    };
    
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
        AbstractDataStorage::ptr storage;

        /**
         * Compares two training examples
         */
        bool operator() (const int lhs, const int rhs) const
        {
            return storage->getDataPoint(lhs)(feature) < storage->getDataPoint(rhs)(feature);
        }
    };
    
    /**
     * This is an ordinary offline decision tree learning algorithm. It learns the
     * tree using the information gain criterion.
     */
    class DecisionTreeLearner : 
            public AbstractDecisionTreeLearner<DecisionTree, DecisionTreeLearnerState>, 
            public Learner<DecisionTree> {
    public:
        DecisionTreeLearner() : AbstractDecisionTreeLearner(),
                smoothingParameter(1),
                useBootstrap(false),
                numBootstrapExamples(1) {}
                
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(DecisionTree::ptr tree, const DecisionTreeLearnerState & state);
                
        /**
         * Actions for the callback function.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_SPLIT_NODE = 2;
        
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
        virtual DecisionTree::ptr learn(AbstractDataStorage::ptr storage);
        
        /**
         * Updates the histograms
         */
        void updateHistograms(DecisionTree::ptr tree, AbstractDataStorage::ptr storage) const;
        
    protected:
        /**
         * The smoothing parameter for the histograms
         */
        float smoothingParameter;
        /**
         * Whether or not bootstrapping shall be used
         */
        bool useBootstrap;
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
            BOOST_ASSERT(_numTrees >= 1);
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
        float getImportance(int feature) const
        {
            return importance[feature];
        }
        
        /**
         * Get the Mean Impurity Decrease variable importance for all features, 
         * see above.
         */
        std::vector<float> & getImportance()
        {
            return importance;
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
        std::vector<float> importance;
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
        static int defaultCallback(RandomForest::ptr forest, const RandomForestLearnerState & state);
        
        /**
         * These are the actions of the learning algorithm that are passed
         * to the callback functions.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_FINISH_TREE = 2;
        const static int ACTION_START_FOREST = 3;
        const static int ACTION_FINISH_FOREST = 4;
        
        RandomForestLearner() : AbstractRandomForestLearner() {}
        
        /**
         * Sets the decision tree learner
         */
        void setTreeLearner(const DecisionTreeLearner & _treeLearner)
        {
            treeLearner = _treeLearner;
        }
        
        /**
         * Returns the decision tree learner
         */
        const DecisionTreeLearner & getTreeLearner() const
        {
            return treeLearner;
        }
        
        /**
         * Returns the decision tree learner
         */
        DecisionTreeLearner & getTreeLearner()
        {
            return treeLearner;
        }
        
        /**
         * Learns a forests. 
         */
        virtual RandomForest::ptr learn(AbstractDataStorage::ptr storage);

    protected:
        /**
         * The tree learner
         */
        DecisionTreeLearner treeLearner;
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
    class BoostedRandomForestLearner : public AbstractLearner<BoostedRandomForest, BoostedRandomForestLearnerState> {
    public:
        
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(BoostedRandomForest::ptr forest, const BoostedRandomForestLearnerState & state);
        
        /**
         * These are the actions of the learning algorithm that are passed
         * to the callback functions.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_FINISH_TREE = 2;
        const static int ACTION_START_FOREST = 3;
        const static int ACTION_FINISH_FOREST = 4;
        
        BoostedRandomForestLearner() : AbstractLearner() {}
        
        /**
         * Sets the decision tree learner
         */
        void setTreeLearner(const DecisionTreeLearner & _treeLearner)
        {
            treeLearner = _treeLearner;
        }
        
        /**
         * Returns the decision tree learner
         */
        const DecisionTreeLearner & getTreeLearner() const
        {
            return treeLearner;
        }
        
        /**
         * Returns the decision tree learner
         */
        DecisionTreeLearner & getTreeLearner()
        {
            return treeLearner;
        }
        
        /**
         * Learns a forests. 
         */
        virtual BoostedRandomForest::ptr learn(AbstractDataStorage::ptr storage);
        
        /**
         * Sets the number of trees. 
         */
        void setNumTrees(int _numTrees)
        {
            BOOST_ASSERT(_numTrees >= 1);
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
        DecisionTreeLearner treeLearner;
    };
}

#endif
