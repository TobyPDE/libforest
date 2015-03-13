#ifndef LIBF_ONLINE_LEARNING_H
#define	LIBF_ONLINE_LEARNING_H

#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>
#include "classifiers.h"
#include "learning.h"

namespace libf {
    /**
     * Forward declarations.
     */
    class DataStorage;
    class DecisionTree;
    class OnlineDecisionTreeLearner;
    class OnlineRandomForestLearner;
    
    /**
     * This is the base class for all online learners.
     */
    template <class T>
    class OnlineLearner {
    public:
        
        /**
         * Learns a classifier.
         */
        virtual T* learn(const DataStorage* storage) = 0;
        
        /**
         * Updates a classifier online.
         */
        virtual T* learn(const DataStorage* storage, T* model) = 0;

    };
    
    class OnlineDecisionTreeLearnerState : public AbstractLearnerState {
    public:
        OnlineDecisionTreeLearnerState() : AbstractLearnerState(),
                node(0),
                objective(0), 
                depth(0) {}
        
        /**
         * Node id.
         */
        int node;
        /**
         * Samples of node.
         */
        int samples;
        /**
         * Objective of splitted node.
         */
        float objective;
        /**
         * Minimum require dobjective.
         */
        float minObjective;
        /**
         * Depth of spitted node.
         */
        int depth;
    };
    
    /**
     * Online decision trees are totally randomized, ie.e. the threshold at each
     * node is chosen randomly. Therefore, the tree has to know the ranges from
     * which to pick the thresholds.
     */
    class RandomThresholdGenerator {
    public:
        
        /**
         * Default constructor: use addFeatureRange to add the value range for
         * an additional feature. Features have to be added in the correct
         * order!
         */
        RandomThresholdGenerator() {};
        
        /**
         * Deduce the feature ranges from the given dataset.
         */
        RandomThresholdGenerator(const DataStorage & storage);
        
        /**
         * Adds a feature range. Note that the features have to be added in the 
         * correct order!
         */
        void addFeatureRange(float _min, float _max)
        {
            min.push_back(_min);
            max.push_back(_max);
        }
        
        /**
         * Adds the same range for num consecutive features.
         * Note that the features have to be added in the correct order!
         */
        void addFeatureRanges(int num, float _min, float _max)
        {
            for (int i = 0; i < num; i++)
            {
                min.push_back(_min);
                max.push_back(_max);
            }
        }
        
        /**
         * Returns minimum for a specific feature.
         */
        float getMin(int feature)
        {
            assert(feature >=0 && feature < getSize());
            return min[feature];
        }
        
        /**
         * Returns maximum for a specific feature.
         */
        float getMax(int feature)
        {
            assert(feature >= 0 && feature < getSize());
            return max[feature];
        }
        
        /**
         * Samples a value uniformly for the given feature.
         */
        float sample(int feature);
        
        /**
         * Returns the size of the generator (number of features).
         */
        int getSize()
        {
            return min.size();
        }
        
    protected:
        /**
         * Minimum value for each feature.
         */
        std::vector<float> min;
        /**
         * Maximum value for each feature.
         */
        std::vector<float> max;
    };
    
    /**
     * Learn a decision tree online, either by passing a single sample at a time
     * or doing online batch learning.
     */
    class OnlineDecisionTreeLearner :
            public AbstractDecisionTreeLearner<DecisionTree, OnlineDecisionTreeLearnerState>,
            public OnlineLearner<DecisionTree> {
    public:
        OnlineDecisionTreeLearner() : AbstractDecisionTreeLearner(),
                smoothingParameter(1),
                useBootstrap(true),
                bootstrapLambda(1.f),
                numThresholds(2*numFeatures),
                minSplitObjective(1.f)
        {
            // Overwrite min split examples.
            minSplitExamples = 30;
            minChildSplitExamples = 15;
        }
        
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(DecisionTree* tree, OnlineDecisionTreeLearnerState* state);
        
        /**
         * Verbose callback for this learner.
         */
        static int verboseCallback(DecisionTree* tree, OnlineDecisionTreeLearnerState* state);
        
        /**
         * Actions for the callback function.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_UPDATE_TREE = 2;
        const static int ACTION_INIT_NODE = 3;
        const static int ACTION_NOT_SPLITTING_NODE = 4;
        const static int ACTION_NOT_SPLITTING_OBJECTIVE_NODE = 5;
        const static int ACTION_SPLIT_NODE = 6;
        
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
         * Sets the minimum objective required for a split.
         */
        void setMinSplitObjective(float _minSplitObjective)
        {
            assert(_minSplitObjective > 0);
            minSplitObjective = _minSplitObjective;
        }
        
        /**
         * Returns the minimum objective required for a split.
         */
        float getMinSplitObjective() const
        {
            return minSplitObjective;
        }
        
        /**
         * Sets the number of thresholds randomly sampled for each node.
         */
        void setNumThresholds(int _numThresholds)
        {
            assert(_numThresholds > 0);
            numThresholds = _numThresholds;
        }
        
        /**
         * Returns the number of thresholds randomly sampled for each node.
         */
        int getNumThresholds() const
        {
            return numThresholds;
        }
        
        /**
         * Sets the threshold generator to use.
         */
        void setThresholdGenerator(RandomThresholdGenerator _thresholdGenerator)
        {
            thresholdGenerator = _thresholdGenerator;
        }
        
        /**
         * Returns the used threshold generator.
         */
        RandomThresholdGenerator & getThresholdGenerator()
        {
            return thresholdGenerator;
        }
        
        /**
         * Learns a decision tree.
         */
        virtual DecisionTree* learn(const DataStorage* storage);
        
        /**
         * Updates the given decision tree on the given data.
         */
        virtual DecisionTree* learn(const DataStorage* storage, DecisionTree* tree);
        
    protected:
        /**
         * For all splits, update left and right child statistics.
         */
        void updateSplitStatistics(std::vector<EfficientEntropyHistogram> & leftChildStatistics, 
                std::vector<EfficientEntropyHistogram> & rightChildStatistics, 
                const std::vector<int> & features,
                const std::vector< std::vector<float> > & thresholds, 
                const DataPoint* x, const int label);
        
        /**
         * The smoothing parameter for the histograms
         */
        float smoothingParameter;
        /**
         * Whether or not bootstrapping shall be used
         */
        bool useBootstrap;
        /**
         * Lambda used for poisson distribution for online bootstrapping.
         */
        float bootstrapLambda;
        /**
         * Number of thresholds randomly sampled. Together with the sampled
         * features these define the tests over which to optimize at
         * each node in online learning.
         * 
         * @see http://lrs.icg.tugraz.at/pubs/saffari_olcv_09.pdf
         */
        int numThresholds;
        /**
         * Minimum objective required for a node to split.
         */
        float minSplitObjective;
        /**
         * The generator to sample random thresholds.
         */
        RandomThresholdGenerator thresholdGenerator;
    };
    
    /**
     * This class holds the current state of the random forest learning
     * algorithm.
     */
    class OnlineRandomForestLearnerState : public AbstractLearnerState {
    public:
        OnlineRandomForestLearnerState() : AbstractLearnerState(),
                tree(0),
                numTrees(0) {}
        
        /**
         * The current tree
         */
        int tree;
        /**
         * Number of learned trees.
         */
        int numTrees;
    };
    
    /**
     * This is an offline random forest learner. 
     */
    class OnlineRandomForestLearner : public AbstractRandomForestLearner<RandomForest, OnlineRandomForestLearnerState>,
            public OnlineLearner<RandomForest> {
    public:
        
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(RandomForest* forest, OnlineRandomForestLearnerState* state);
        
        /**
         * Verbose callback for this learner.
         */
        static int verboseCallback(RandomForest* forest, OnlineRandomForestLearnerState* state);
         
        /**
         * These are the actions of the learning algorithm that are passed
         * to the callback functions.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_FINISH_TREE = 2;
        const static int ACTION_START_FOREST = 3;
        const static int ACTION_FINISH_FOREST = 4;
        
        OnlineRandomForestLearner() : AbstractRandomForestLearner(),
                treeLearner(0) {}
        
        /**
         * Sets the decision tree learner
         */
        void setTreeLearner(OnlineDecisionTreeLearner* _treeLearner)
        {
            treeLearner = _treeLearner;
        }
        
        /**
         * Returns the decision tree learner
         */
        OnlineDecisionTreeLearner* getTreeLearner() const
        {
            return treeLearner;
        }
        
        /**
         * Learns a decision forest.
         */
        virtual RandomForest* learn(const DataStorage* storage);
        
        /**
         * Learns a forests. 
         */
        virtual RandomForest* learn(const DataStorage* storage, RandomForest* forest);

    protected:
        /**
         * The tree learner
         */
        OnlineDecisionTreeLearner* treeLearner;
    };
}

#endif	/* LIBF_ONLINE_LEARNING_H */

