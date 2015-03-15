#ifndef LIBF_UNSUPERVISED_LEARNING_H
#define	LIBF_UNSUPERVISED_LEARNING_H

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
    class UnlabeledDataStorage;
    class DensityDecisionTree;
    class DensityDecisionTreeLearner;
    
    /**
     * This is the base class for all unsupervised learners.
     */
    template <class T>
    class UnsupervisedLearner {
    public:
        
        /**
         * Learns a classifier in an unsupervised fashion.
         */
        virtual std::shared_ptr<T> learn(AbstractDataStorage::ptr storage) = 0;

    };
    
    class DensityDecisionTreeLearnerState : public AbstractLearnerState {
    public:
        DensityDecisionTreeLearnerState() : 
            node(0),
            samples(0),
            objective(0) {};
            
            /**
             * Current node.
             */
            int node;
            /**
             * Depth of node.
             */
            int depth;
            /**
             * Number of samples at current node.
             */
            int samples;
            /**
             * Objective of split or not split.
             */
            float objective;
    };
    
    class DensityDecisionTreeLearner : 
            public AbstractDecisionTreeLearner<DensityDecisionTree, DensityDecisionTreeLearnerState>,
            public UnsupervisedLearner<DensityDecisionTree> {
    public:
        DensityDecisionTreeLearner() : AbstractDecisionTreeLearner() {};
                
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(DensityDecisionTree::ptr tree, const DensityDecisionTreeLearnerState & state);
                
        /**
         * Actions for the callback function.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_SPLIT_NODE = 2;
        
        /**
         * Learn a density tree.
         */
        DensityDecisionTree::ptr learn(AbstractDataStorage::ptr storage);
        
    protected:
        void updateLeafNodeGaussian(Gaussian gaussian, EfficientCovarianceMatrix covariance);
        
    };
}

#endif	/* LIBF_UNSUPERVISED_LEARNING_H */

