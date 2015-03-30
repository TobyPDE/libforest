#ifndef LIBF_CLASSIFIERLEARNINGOFFLINE_H
#define LIBF_CLASSIFIERLEARNINGOFFLINE_H

#include <functional>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <iomanip>
#include <random>
#include <type_traits>

#include "error_handling.h"
#include "data.h"
#include "classifier.h"
#include "learning.h"
#include "learning_tools.h"

namespace libf {
    /**
     * This is the base class for all tree classifer learners. It includes 
     * parameter settings all learners have in common. 
     */
    class AbstractTreeClassifierLearner : public AbstractTreeLearner {
    public:
        
        AbstractTreeClassifierLearner() : 
                smoothingParameter(1),
                useBootstrap(false),
                numBootstrapExamples(1) {}
        
        /**
         * Sets the smoothing parameter. The smoothing parameter is the value 
         * the histograms at the leaf nodes are initialized with. 
         * 
         * @param _smoothingParameter The smoothing parameter
         */
        void setSmoothingParameter(float _smoothingParameter)
        {
            smoothingParameter = _smoothingParameter;
        }
        
        /**
         * Returns the smoothing parameter. 
         * 
         * @return The smoothing parameter
         */
        float getSmoothingParameter() const
        {
            return smoothingParameter;
        }
        
        /**
         * Sets whether or not bootstrapping shall be used. If true, then the
         * tree is learned on an iid sampled subset of the training data. 
         * 
         * @param _useBootstrap If true, the bootstrap sampling is performed
         */
        void setUseBootstrap(bool _useBootstrap)
        {
            useBootstrap = _useBootstrap;
        }

        /**
         * Returns whether or not bootstrapping is used
         * 
         * @return True if bootstrap sampling is performed
         */
        bool getUseBootstrap() const
        {
            return useBootstrap;
        }
        
        /**
         * Sets the number of samples to use for bootstrapping.
         * 
         * @param _numBootstrapExamples The number of bootstrap samples
         */
        void setNumBootstrapExamples(int _numBootstrapExamples)
        {
            numBootstrapExamples = _numBootstrapExamples;
        }
        
        /**
         * Returns the number of samples used for bootstrapping.
         * 
         * @return The number of bootstrap samples
         */
        int getNumBootstrapExamples() const
        {
            return numBootstrapExamples;
        }
        
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
     * This is an ordinary offline decision tree learning algorithm. It learns the
     * tree using the information gain criterion.
     */
    class DecisionTreeLearner : 
            public AbstractTreeClassifierLearner, 
            public OfflineLearnerInterface<DecisionTree> {
    public:
        
        /**
         * This is the learner state for the GUI
         */
        typedef TreeLearnerState State;
            
        DecisionTreeLearner() : AbstractTreeClassifierLearner() {}
        
        /**
         * Learns a decision tree on a data set.
         * 
         * @param storage The training set
         * @param state The learning state
         * @return The learned tree
         */
        virtual DecisionTree::ptr learn(AbstractDataStorage::ptr storage, State & state);
        
        /**
         * Learns a decision tree on a data set.
         * 
         * @param storage The training set
         * @return The learned tree
         */
        virtual DecisionTree::ptr learn(AbstractDataStorage::ptr storage)
        {
            // Just create a state and don't do anything with it
            State state;
            return this->learn(storage, state);
        }
    };
    
    /**
     * This is a projective decision tree learning algorithm. It learns the
     * tree using the information gain criterion.
     */
    class ProjectiveDecisionTreeLearner : 
            public AbstractTreeClassifierLearner, 
            public OfflineLearnerInterface<ProjectiveDecisionTree> {
    public:
        ProjectiveDecisionTreeLearner() : AbstractTreeClassifierLearner() {}
        
        /**
         * This is the learner state for the GUI
         */
        typedef TreeLearnerState State;
        
        /**
         * Learns a decision tree on a data set.
         * 
         * @param storage The training set
         * @param state The learning state
         * @return The learned tree
         */
        virtual ProjectiveDecisionTree::ptr learn(AbstractDataStorage::ptr storage, State & state);
        
        /**
         * Learns a decision tree on a data set.
         * 
         * @param storage The training set
         * @return The learned tree
         */
        virtual ProjectiveDecisionTree::ptr learn(AbstractDataStorage::ptr storage)
        {
            // Just create a state and don't do anything with it
            State state;
            return this->learn(storage, state);
        }
    };
    
    /**
     * Learn a decision tree online, either by passing a single sample at a time
     * or doing online batch learning.
     */
    class OnlineDecisionTreeLearner :
            public AbstractTreeClassifierLearner, 
            public OnlineLearnerInterface<OnlineDecisionTree> {
    public:
        
        /**
         * This is the learner state for the GUI
         */
        typedef TreeLearnerState State;
        
        OnlineDecisionTreeLearner() : AbstractTreeClassifierLearner(),
                bootstrapLambda(1.f),
                numThresholds(2*numFeatures),
                minSplitObjective(1.f)
        {
            // Overwrite min split examples.
            minSplitExamples = 30;
            minChildSplitExamples = 15;
        }
        
        /**
         * Sets the minimum objective required for a split.
         * 
         * @param _minSplitObjective The new split objective
         */
        void setMinSplitObjective(float _minSplitObjective)
        {
            BOOST_ASSERT(_minSplitObjective > 0);
            minSplitObjective = _minSplitObjective;
        }
        
        /**
         * Returns the minimum objective required for a split.
         * 
         * @return The min split objective
         */
        float getMinSplitObjective() const
        {
            return minSplitObjective;
        }
        
        /**
         * Sets the number of thresholds randomly sampled for each node.
         * 
         * @param _numThresholds The number of thresholds
         */
        void setNumThresholds(int _numThresholds)
        {
            BOOST_ASSERT(_numThresholds > 0);
            numThresholds = _numThresholds;
        }
        
        /**
         * Returns the number of thresholds randomly sampled for each node.
         * 
         * @return The number of thresholds
         */
        int getNumThresholds() const
        {
            return numThresholds;
        }
        
        /**
         * Sets the threshold generator to use.
         */
        void setThresholdGenerator(RandomThresholdGenerator & _thresholdGenerator)
        {
            thresholdGenerator = _thresholdGenerator;
        }
        
        /**
         * Returns the used threshold generator.
         * 
         * @return A reference to the random threshold generator
         */
        RandomThresholdGenerator & getThresholdGenerator()
        {
            return thresholdGenerator;
        }
        
        /**
         * Returns the used threshold generator.
         * 
         * @return A const reference to the random threshold generator
         */
        const RandomThresholdGenerator & getThresholdGenerator() const
        {
            return thresholdGenerator;
        }
        
        /**
         * Updates the given decision tree on the given data.
         * 
         * @param storage The data to train on
         * @param tree The tree to update
         * @param state The learner state
         * @return The learned decision tree
         */
        virtual OnlineDecisionTree::ptr learn(AbstractDataStorage::ptr storage, OnlineDecisionTree::ptr tree, State & state);
        
        /**
         * Learns a decision tree.
         * 
         * @param storage The data to train on
         * @return The learned decision tree
         */
        virtual OnlineDecisionTree::ptr learn(AbstractDataStorage::ptr storage)
        {
            State state;
            auto tree = std::make_shared<OnlineDecisionTree>();
            tree->addNode();
            return learn(storage, tree, state);
        }
        
        /**
         * Learns a decision tree.
         * 
         * @param storage The data to train on
         * @param state The learner state
         * @return The learned decision tree
         */
        virtual OnlineDecisionTree::ptr learn(AbstractDataStorage::ptr storage, State & state)
        {
            auto tree = std::make_shared<OnlineDecisionTree>();
            tree->addNode();
            return learn(storage, tree, state);
        }
        
        /**
         * Learns a decision tree.
         * 
         * @param storage The data to train on
         * @param tree The tree to update
         * @return The learned decision tree
         */
        virtual OnlineDecisionTree::ptr learn(AbstractDataStorage::ptr storage, OnlineDecisionTree::ptr tree)
        {
            State state;
            return learn(storage, tree, state);
        }
        
    protected:
        /**
         * For all splits, update left and right child statistics.
         */
        void updateSplitStatistics(std::vector<EfficientEntropyHistogram> & leftChildStatistics, 
                std::vector<EfficientEntropyHistogram> & rightChildStatistics, 
                const std::vector<int> & features,
                const std::vector< std::vector<float> > & thresholds, 
                const DataPoint & x, const int label);
        
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
     * This is an offline random forest learner. T is the classifier learner. 
     */
    template <class L>
    class RandomForestLearner : 
            public AbstractForestLearner,
            public OfflineLearnerInterface< RandomForest<typename L::HypothesisType> > {
    public:
        
        /**
         * The state type for this learner
         */
        typedef RandomForestLearnerState<L> State;
        
        /**
         * Returns the decision tree learner
         */
        const L & getTreeLearner() const
        {
            return treeLearner;
        }
        
        /**
         * Returns the decision tree learner
         */
        L & getTreeLearner()
        {
            return treeLearner;
        }
        
        /**
         * Learns a forests. 
         */
        virtual typename RandomForest<typename L::HypothesisType>::ptr learn(AbstractDataStorage::ptr storage, State & state)
        {
            // Set up the state for the call backs
            state.reset();
            state.started = true;
            state.total = this->getNumTrees();
            state.treeLearnerStates.resize(this->getNumThreads());
            
            // Set up the empty random forest
            auto forest = ForestFactory< RandomForest<typename L::HypothesisType> >::create();

            #pragma omp parallel for num_threads(this->numThreads)
            for (int i = 0; i < this->getNumTrees(); i++)
            {
                #pragma omp critical
                {
                    state.startedProcessing++;
                }
                
                // Learn the tree
#if LIBF_ENABLE_OPENMP
                auto tree = treeLearner.learn(storage, state.treeLearnerStates[omp_get_thread_num()]);
#else
                auto tree = treeLearner.learn(storage, state.treeLearnerStates[0]);
#endif
                
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
        virtual typename RandomForest<typename L::HypothesisType>::ptr learn(AbstractDataStorage::ptr storage)
        {
            State state;
            return this->learn(storage, state);
        }
        
    protected:
        /**
         * The tree learner
         */
        L treeLearner;
    };
    
    /**
     * This is an online random forest learner. T is the classifier learner. 
     */
    template <class L>
    class OnlineRandomForestLearner : 
            public AbstractForestLearner,
            public OnlineLearnerInterface< RandomForest<typename L::HypothesisType> > {
    public:
        
        /**
         * The state type for this learner
         */
        typedef RandomForestLearnerState<L> State;
        
        /**
         * Returns the decision tree learner
         */
        const L & getTreeLearner() const
        {
            return treeLearner;
        }
        
        /**
         * Returns the decision tree learner
         */
        L & getTreeLearner()
        {
            return treeLearner;
        }
        
        /**
         * Updates an already learned classifier. 
         * 
         * @param storage The storage to train the classifier on
         * @param classifier The base classifier
         * @param state The state variable for this learner
         * @return The trained classifier
         */
        virtual typename RandomForest<typename L::HypothesisType>::ptr learn(
                AbstractDataStorage::ptr storage, 
                typename RandomForest<typename L::HypothesisType>::ptr forest, 
                State & state)
        {
            // Set up the state for the call backs
            state.reset();
            state.started = true;
            state.total = this->getNumTrees();
            state.treeLearnerStates.resize(this->getNumThreads());
            
            // Add the required number of trees if there are too few trees in 
            // the forest
            for (int i = forest->getSize(); i < this->getNumTrees(); i++)
            {
                auto tree = TreeFactory<typename L::HypothesisType>::create();
                forest->addTree(tree);
            }
            
            #pragma omp parallel for num_threads(this->numThreads)
            for (int i = 0; i < this->numTrees; i++)
            {
                #pragma omp critical
                {
                    state.startedProcessing++;
                }

                auto tree = forest->getTree(i);
#if LIBF_ENABLE_OPENMP
                this->treeLearner.learn(storage, tree, state.treeLearnerStates[omp_get_thread_num()]);
#else
                this->treeLearner.learn(storage, tree, state.treeLearnerStates[0]);
#endif
                
                #pragma omp critical
                {
                    state.processed++;
                }
            }

            state.terminated = true;
            
            return forest;
        }
        
        /**
         * Learns an online forests. 
         * 
         * @param storage The data to train the forest on
         */
        virtual typename RandomForest<typename L::HypothesisType>::ptr learn(AbstractDataStorage::ptr storage)
        {
            State state;
            auto forest = ForestFactory<RandomForest<typename L::HypothesisType> >::create();
            return learn(storage, forest, state);
        }
        
        /**
         * Updates an already learned classifier. 
         * 
         * @param storage The storage to train the classifier on
         * @param classifier The base classifier
         * @return The trained classifier
         */
        virtual typename RandomForest<typename L::HypothesisType>::ptr learn(
                AbstractDataStorage::ptr storage, 
                typename RandomForest<typename L::HypothesisType>::ptr forest)
        {
            State state;
            return learn(storage, forest, state);
        }
        
        /**
         * Learns an online forests. 
         * 
         * @param storage The data to train the forest on
         * @param state The state variable for this learner
         */
        virtual typename RandomForest<typename L::HypothesisType>::ptr learn(AbstractDataStorage::ptr storage, State & state)
        {
            auto forest = ForestFactory<RandomForest<typename L::HypothesisType> >::create();
            return learn(storage, forest, state);
        }

    protected:
        /**
         * The tree learner
         */
        L treeLearner;
    };
    
    /**
     * This is a random forest learner. 
     */
    template <class L>
    class BoostedRandomForestLearner : 
            public AbstractForestLearner, 
            public OfflineLearnerInterface< BoostedRandomForest<typename L::HypothesisType> > {
    public:
        
        /**
         * The state type for this learner
         */
        class State : public ForestLearnerState {
        public:
            
            State() : lastAlpha(0), lastError(0) {}
            
            /**
             * Prints the state into the console. 
             */
            virtual void print() const
            {
                ForestLearnerState::print();
                std::cout << "Alpha: " << lastAlpha << std::endl;
                std::cout << "Error: " << lastError << std::endl;
                treeLearnerState.print();
            }

            /**
             * Resets the state
             */
            virtual void reset()
            {
                ForestLearnerState::reset();
                lastAlpha = 0;
                lastError = 0;
            }
            
            /**
             * The last alpha value
             */
            float lastAlpha;
            /**
             * The last error value
             */
            float lastError;
            /**
             * The state of the tree learner
             */
            typename L::State treeLearnerState;
        };
        
        /**
         * Returns the decision tree learner
         */
        const L & getTreeLearner() const
        {
            return treeLearner;
        }
        
        /**
         * Returns the decision tree learner
         */
        L & getTreeLearner()
        {
            return treeLearner;
        }
        
        /**
         * Learns a forests. 
         */
        virtual typename BoostedRandomForest<typename L::HypothesisType>::ptr learn(AbstractDataStorage::ptr storage, State & state)
        {
            // Set up the state for the call backs
            state.reset();
            state.total = this->getNumTrees();
            state.started = true;
            
            // Set up the empty random forest
             auto forest = std::make_shared< BoostedRandomForest<typename L::HypothesisType> >();

            // Set up the weights for the data points
            const int N = storage->getSize();
            std::vector<float> dataWeights(N);
            std::vector<float> cumsum(N);
            std::vector<bool> misclassified(N);
            for (int n = 0; n < N; n++)
            {
                dataWeights[n] = 1.0f/N;
                cumsum[n] = (n+1) * 1.0f/N;
                misclassified[n] = false;
            }

            // We need this distribution in order to sample according to the weights
            std::random_device rd;
            std::mt19937 g(rd());
            std::uniform_real_distribution<float> U(0, 1);

            const int C = storage->getClasscount();

            for (int i = 0; i < this->numTrees; i++)
            {
                state.startedProcessing++;

                // Learn the tree
                // --------------

                // Sample data points according to the weights
                ReferenceDataStorage::ptr treeData = std::make_shared<ReferenceDataStorage>(storage);

                for (int n = 0; n < N; n++)
                {
                    const float u = U(g);
                    int index = 0;
                    while (u > cumsum[index] && index < N-1)
                    {
                        index++;
                    }
                    treeData->addDataPoint(index);
                }

                // Learn the tree
                auto tree = treeLearner.learn(treeData, state.treeLearnerState);

                // Calculate the error term
                float error = 0;
                for (int n = 0; n < N; n++)
                {
                    const int predictedLabel = tree->classify(storage->getDataPoint(n));
                    if (predictedLabel != storage->getClassLabel(n))
                    {
                        error += dataWeights[n];
                        misclassified[n] = true;
                    }
                    else
                    {
                        misclassified[n] = false;
                    }
                }

                // Compute the classifier weight
                const float alpha = std::log((1-error)/error) + std::log(C - 1);

                // Update the weights
                float total = 0;
                for (int n = 0; n < N; n++)
                {
                    if (misclassified[n])
                    {
                        dataWeights[n] *= std::exp(alpha);
                    }
                    total += dataWeights[n];
                }
                dataWeights[0] /= total;
                cumsum[0] = dataWeights[0];
                for (int n = 1; n < N; n++)
                {
                    dataWeights[n] /= total;
                    cumsum[n] = dataWeights[n] + cumsum[n-1];
                }

                // Create the weak classifier
                auto weakClassifier = std::make_shared<WeakClassifier<typename L::HypothesisType> >();
                weakClassifier->setClassifier(tree);
                weakClassifier->setWeight(alpha);
                
                // Add the classifier
                forest->addTree(weakClassifier);

                // --------------
                // Add it to the forest
                state.processed++;
                state.lastAlpha = alpha;
                state.lastError = error;
            }

            state.terminated = true;
            
            return forest;
        }
        
        /**
         * Learns a forests. 
         */
        virtual typename BoostedRandomForest<typename L::HypothesisType>::ptr learn(AbstractDataStorage::ptr storage)
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
}

#endif
