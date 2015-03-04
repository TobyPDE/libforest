#ifndef LIBF_ONLINE_LEARNING_H
#define	LIBF_ONLINE_LEARNING_H

#include <cassert>
#include <functional>
#include <iostream>
#include <chrono>
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
    
    /**
     * This is the base class for all online learners.
     */
    template <class T>
    class OnlineLearner {
    public:
        
        /**
         * Learns a classifier online (updates a given classifier).
         */
        virtual T* learn(const DataStorage* storage, T* model = 0) = 0;

    };
    
    class OnlineDecisionTreeLearnerState {
    public:
        OnlineDecisionTreeLearnerState() : 
                action(0), 
                learner(0), 
                tree(0),
                node(0),
                objective(0), 
                depth(0),
                startTime(std::chrono::high_resolution_clock::now()) {}
        
        /**
         * The current action
         */
        int action;
        /**
         * The learner object
         */
        const OnlineDecisionTreeLearner* learner;
        /**
         * The learned object
         */
        const DecisionTree* tree;
        /**
         * Node id.
         */
        int node;
        /**
         * Objective of splitted node.
         */
        float objective;
        /**
         * Depth of spitted node.
         */
        int depth;
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
    
    class OnlineDecisionTreeLearner : public AbstractDecisionTreeLearner<OnlineDecisionTreeLearnerState>,
            public OnlineLearner<DecisionTree> {
    public:
        OnlineDecisionTreeLearner() : AbstractDecisionTreeLearner(), 
                numThresholds(25),
                minSplitObjective(0.1f) {};
        
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(DecisionTree* tree, OnlineDecisionTreeLearnerState* state);
                
        /**
         * Actions for the callback function.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_UPDATE_TREE = 2;
        const static int ACTION_INIT_NODE = 3;
        const static int ACTION_UPDATE_NODE = 4;
        const static int ACTION_SPLIT_NODE = 5;
                
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
         * Configures the learning automatically depending on a certain data
         * set. 
         */
        virtual void autoconf();
        
        /**
         * Updates the given decision tree on the given data.
         */
        virtual DecisionTree* learn(const DataStorage* storage, DecisionTree* tree);
        
        /**
         * Dumps the settings
         */
        virtual void dumpSetting(std::ostream & stream = std::cout) const;
        
    protected:
        /**
         * For all splits, update left and right child statistics.
         */
        void updateSplitStatistics(std::vector<EfficientEntropyHistogram> leftChildStatistics, 
                std::vector<EfficientEntropyHistogram> rightChildStatistics, 
                const std::vector<int> & features,
                const std::vector< std::vector<float> > & thresholds, 
                const std::pair<DataPoint*, int> & x);
        
        /**
         * Minimum objective required for a node to split.
         */
        float minSplitObjective;
        /**
         * Number of thresholds randomly sampled. Together with the sampled
         * features these define the tests over which to optimize at
         * each node in online learning.
         * 
         * @see http://lrs.icg.tugraz.at/pubs/saffari_olcv_09.pdf
         */
        int numThresholds;
    };
}

#endif	/* LIBF_ONLINE_LEARNING_H */

