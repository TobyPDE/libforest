#ifndef LIBF_UNSUPERVISED_LEARNING_H
#define	LIBF_UNSUPERVISED_LEARNING_H

#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>
#include "estimators.h"
#include "learning.h"

namespace libf {
    /**
     * Forward declarations.
     */
    class UnlabeledDataStorage;
    class DensityTree;
    class DensityTreeLearner;
    
    /**
     * This is the base class for all unsupervised learners.
     */
    template <class T>
    class UnsupervisedLearner {
    public:
        
        /**
         * Learns a classifier in an unsupervised fashion.
         */
        virtual T* learn(const UnlabeledDataStorage* storage) = 0;

    };
    
    class DensityTreeLearnerState : public AbstractLearnerState {
    public:
        DensityTreeLearnerState() : 
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
    
    class DensityTreeLearner : 
            public AbstractDecisionTreeLearner<DensityTree, DensityTreeLearnerState>,
            public UnsupervisedLearner<DensityTree> {
    public:
        DensityTreeLearner() : AbstractDecisionTreeLearner() {};
                
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(DensityTree* tree, DensityTreeLearnerState* state);
                
        /**
         * Actions for the callback function.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_SPLIT_NODE = 2;
        
        /**
         * Learn a density tree.
         */
        DensityTree* learn(const UnlabeledDataStorage* storage);
        
    protected:
        void updateLeafNodeGaussian(Gaussian gaussian, EfficientCovarianceMatrix covariance);
        
    };
}

#endif	/* LIBF_UNSUPERVISED_LEARNING_H */

