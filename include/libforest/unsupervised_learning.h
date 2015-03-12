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
             * Max depth allowed.
             */
            int maxDepth;
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
         * Verbose callback for this learner.
         */
        static int verboseCallback(DensityTree* forest, DensityTreeLearnerState* state);   
        
        /**
         * Actions for the callback function.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_SPLIT_NODE = 2;
        const static int ACTION_PROCESS_NODE = 3;
        const static int ACTION_INIT_NODE = 4;
        const static int ACTION_NOT_SPLIT_NODE = 5;
        const static int ACTION_NO_SPLIT_NODE = 6;
        
        /**
         * Learn a density tree.
         */
        DensityTree* learn(const UnlabeledDataStorage* storage);

    private:
        void updateLeafNodeGaussian(Gaussian & gaussian, EfficientCovarianceMatrix & covariance);
        
    };
    
    /**
     * This class holds the current state of the random forest learning
     * algorithm.
     */
    class DensityForestLearnerState : public AbstractLearnerState {
    public:
        DensityForestLearnerState() : AbstractLearnerState(),
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
    class DensityForestLearner : public AbstractRandomForestLearner<DensityForest, DensityForestLearnerState>,
            public UnsupervisedLearner<DensityForest> {
    public:
        
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(DensityForest* forest, DensityForestLearnerState* state);
         
        /**
         * These are the actions of the learning algorithm that are passed
         * to the callback functions.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_FINISH_TREE = 2;
        const static int ACTION_START_FOREST = 3;
        const static int ACTION_FINISH_FOREST = 4;
        
        DensityForestLearner() : AbstractRandomForestLearner(),
                treeLearner(0) {}
        
        /**
         * Sets the decision tree learner
         */
        void setTreeLearner(DensityTreeLearner* _treeLearner)
        {
            treeLearner = _treeLearner;
        }
        
        /**
         * Returns the decision tree learner
         */
        DensityTreeLearner* getTreeLearner() const
        {
            return treeLearner;
        }
        
        /**
         * Learns a forests. 
         */
        virtual DensityForest* learn(const UnlabeledDataStorage* storage);

    protected:
        /**
         * The tree learner
         */
        DensityTreeLearner* treeLearner;
    };
}

#endif	/* LIBF_UNSUPERVISED_LEARNING_H */

