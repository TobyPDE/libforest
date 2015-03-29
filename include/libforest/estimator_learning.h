#ifndef LIBF_UNSUPERVISED_LEARNING_H
#define	LIBF_UNSUPERVISED_LEARNING_H

#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>
#include "estimator.h"
#include "learning.h"

namespace libf {
    /**
     * Forward declarations.
     */
    class DensityTree;
    class DensityTreeLearner;
    
    /**
     * Learnes a density tree on a given unlabeled dataset.
     */
    class DensityTreeLearner : 
            public AbstractTreeLearner,
            public OfflineLearnerInterface<DensityTree> {
    public:
        /**
         * The state type for this learner
         */
        using State = TreeLearnerState;
        
        /**
         * Learn a density tree.
         */
        DensityTree::ptr learn(AbstractDataStorage::ptr storage, State & state);
        
        /**
         * Learn a density tree.
         */
        DensityTree::ptr learn(AbstractDataStorage::ptr storage)
        {
            State state;
            return learn(storage, state);
        }

    private:
        /**
         * Updates the leaf node Gaussian estimates given the current covariance
         * and mean estimate.
         */
        void updateLeafNodeGaussian(Gaussian & gaussian, EfficientCovarianceMatrix & covariance);
        
    };
    
    /**
     * This is an offline density forest learner.
     */
    template <class L>
    class DensityForestLearner : 
            public AbstractForestLearner,
            public OfflineLearnerInterface<DensityForest<typename L::HypothesisType> > {
    public:
        
        /**
         * The state type for this learner
         */
        using State = RandomForestLearnerState<L>;
        
        /**
         * Returns the decision tree learner
         * 
         * @return A reference to the learner
         */
        L & getTreeLearner()
        {
            return treeLearner;
        }
        
        
        /**
         * Returns the decision tree learner
         * 
         * @return A reference to the learner
         */
        const L & getTreeLearner() const
        {
            return treeLearner;
        }
        
        /**
         * Learns a forests. 
         */
        virtual std::shared_ptr<DensityForest<typename L::HypothesisType> > learn(AbstractDataStorage::ptr storage, State & state)
        {
            // Set up the state for the call backs
            state.reset();
            state.started = true;
            state.total = this->getNumTrees();
            state.treeLearnerStates.resize(this->getNumThreads());
            
            // Set up the empty random forest
            auto forest = ForestFactory< DensityForest<typename L::HypothesisType> >::create();

            #pragma omp parallel for num_threads(this->numThreads)
            for (int i = 0; i < this->getNumTrees(); i++)
            {
                #pragma omp critical
                {
                    state.startedProcessing++;
                }
                
                // Learn the tree
                //auto tree = treeLearner.learn(storage, state.treeLearnerStates[omp_get_thread_num()]);
                auto tree = treeLearner.learn(storage, state.treeLearnerStates[0]);
                
                // Add it to the forest
                #pragma omp critical
                {
                    state.processed++;
                    forest->addTree(tree);
                }
            }
            
            state.terminated = true;
            
            return forest;
        }
        
        /**
         * Learns a forests. 
         */
        virtual std::shared_ptr<DensityForest<typename L::HypothesisType> > learn(AbstractDataStorage::ptr storage)
        {
            State state;
            return learn(storage, state);
        }

    protected:
        /**
         * The tree learner
         */
        L treeLearner;
    };
    
    /**
     * Learns a kernel density tree.
     */
    class KernelDensityTreeLearner : 
            public AbstractTreeLearner,
            public OfflineLearnerInterface<KernelDensityTree> {
    public:

        /**
         * The state type for this learner
         */
        using State = TreeLearnerState;
        
        /**
         * Constructs a kernel density tree learner.
         */
        KernelDensityTreeLearner() : 
                // TODO: There is a memory leak
                kernel(new MultivariateGaussianKernel()), 
                bandwidthSelectionMethod(KernelDensityEstimator::BANDWIDTH_RULE_OF_THUMB) {};
        
        /**
         * Sets the kernel to use.
         */
        void setKernel(MultivariateKernel* _kernel)
        {
            kernel = _kernel;
        }
        
        /**
         * Returns the used kernel.
         */
        MultivariateKernel* getKernel()
        {
            return kernel;
        }
               
        /**
         * Sets the bandwidth selection method to use.
         */
        void setBandwidthSelectionMethod(int _bandwidthSelectionMethod)
        {
            bandwidthSelectionMethod = _bandwidthSelectionMethod;
        }
        
        /**
         * Returns the bandwidth selection method to use.
         */
        int getBandwidthSelectionMethod()
        {
            return bandwidthSelectionMethod;
        }
        
        /**
         * Learns a kernel density tree.
         */
        virtual KernelDensityTree::ptr learn(AbstractDataStorage::ptr storage, State & state);
        
        /**
         * Learns a kernel density tree.
         */
        virtual KernelDensityTree::ptr learn(AbstractDataStorage::ptr storage)
        {
            State state;
            return learn(storage, state);
        }
        
    private:
        /**
         * Initializes the leaf node estimator.
         */
        void initializeLeafNodeEstimator(Gaussian & gaussian,
                EfficientCovarianceMatrix & covariance,
                KernelDensityEstimator & estimator,
                const std::vector<int> & trainingExamples, 
                AbstractDataStorage::ptr storage);
        
        /**
         * Each leaf node has an associated kernel density estimator.
         */
        std::vector<KernelDensityEstimator> kernelEstimators;
        /**
         * Kernel used at the leaf node.
         */
        MultivariateKernel* kernel;
        /**
         * Bandwidth selection method to use in leaf nodes.
         */
        int bandwidthSelectionMethod;
        
    };
}

#endif	/* LIBF_UNSUPERVISED_LEARNING_H */

