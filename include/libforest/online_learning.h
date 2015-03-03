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
    class RandomForest;
    
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
    
    class OnlineDecisionTreeLearner : public AbstractDecisionTreeLearner<DecisionTreeLearnerState>,
            public OnlineLearner<DecisionTree> {
    public:
        OnlineDecisionTreeLearner() : AbstractDecisionTreeLearner(), 
                numThresholds(25),
                minSplitObjective(0.1f) {};
        
        /**
         * Sets the minimum objective required for a split.
         */
        void setMinSplitObjective(float _minSplitObjective)
        {
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
        virtual void autoconf(const DataStorage* storage);
        
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

