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
        virtual T* learn(const UnlabeledDataStorage* storage) = 0;

    };
    
    class DensityDecisionTreeLearnerState : public AbstractLearnerState {
        DensityDecisionTreeLearnerState() : 
            node(0),
            samples(0),
            objective(0) {};
            
            /**
             * Current node.
             */
            int node;
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
            public AbstractDecisionTreeLearner<DensityDecisionTreeLearnerState>,
            public UnsupervisedLearner<DensityDecisionTree> {
    public:
        DensityDecisionTreeLearner() : 
                maxDepth(0), 
                minSplitExamples(0), 
                minChildSplitExamples(0) {};
                
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(KernelDecisionTree* tree, DensityDecisionTreeLearnerState* state);
                
        /**
         * Actions for the callback function.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_SPLIT_NODE = 2;
        
        /**
         * Learn a density tree.
         */
        DensityDecisionTree* learn(const UnlabeledDataStorage* storage);
        
    protected:
        void updateGaussian();
        
        /**
         * Maximum depth of the tree.
         */
        int maxDepth;
        /**
         * Minimum examples required for a split.
         */
        int minSplitExamples;
        /**
         * Minimum examples required in the children of each node.
         */
        int minChildSplitExamples;
    };
}

#endif	/* LIBF_UNSUPERVISED_LEARNING_H */

