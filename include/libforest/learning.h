#ifndef LIBF_LEARNING_H
#define LIBF_LEARNING_H

#include <functional>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <iomanip>
#include <random>
#include <thread>
#include <type_traits>
#include "error_handling.h"
#include "data.h"
#include "classifier.h"

namespace libf {
    /**
     * Abstract learner state for measuring time and defining actions.
     */
    class AbstractLearnerState {
    public:
        AbstractLearnerState() : 
                action(0),
                startTime(std::chrono::high_resolution_clock::now()), 
                terminated(false), 
                started(false) {}
        
        /**
         * The current action
         * TODO: Remove this field
         */
        int action;
        /**
         * The start time
         */
        std::chrono::high_resolution_clock::time_point startTime;
        /**
         * Whether the learning has terminated
         */
        bool terminated;
        /**
         * Whether the learning has started yet
         */
        bool started;
        
        /**
         * Returns the passed time in microseconds
         * 
         * @return The time passed since instantiating the state object
         */
        std::chrono::microseconds getPassedTime() const
        {
            const std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>( now - startTime );
        }
        
        /**
         * Returns the passed time in seconds
         * 
         * @return The time passed since instantiating the state object
         */
        float getPassedTimeInSeconds() const
        {
            return static_cast<float>(getPassedTime().count()/1000000);
        }
        
        /**
         * Prints the state into the console. 
         */
        virtual void print() const = 0;
        
        /**
         * Resets the state
         */
        virtual void reset() 
        {
            started = false;
            terminated = false;
            startTime = std::chrono::high_resolution_clock::now();
        }
    };
    
    /**
     * This class includes the basic state variables of any tree learner. 
     */
    class TreeLearnerState : public AbstractLearnerState {
    public:
        TreeLearnerState() : 
                AbstractLearnerState(), 
                total(0), 
                processed(0), 
                depth(0), 
                numNodes(0) {}

        /**
         * The total number of training examples
         */
        int total;
        /**
         * The number of processed examples
         */
        int processed;
        /**
         * The depth of the tree
         */
        int depth;
        /**
         * The total number of nodes
         */
        int numNodes;
        
        /**
         * Prints the state into the console. 
         */
        virtual void print() const;
        
        /**
         * Resets the state
         */
        virtual void reset();
    };
    
    /**
     * This class includes the basic information every forest learner has. 
     */
    class ForestLearnerState : public AbstractLearnerState {
    public:
        ForestLearnerState() : 
                AbstractLearnerState(), 
                total(0), 
                startedProcessing(0), 
                processed(0) {}

        /**
         * The total number of trees to train
         */
        int total;
        /**
         * The number of trees started
         */
        int startedProcessing;
        /**
         * The number of finished trees
         */
        int processed;
        
        /**
         * Prints the state into the console. 
         */
        virtual void print() const;
        
        /**
         * Resets the state
         */
        virtual void reset();
    };
    
    
    /**
     * This class includes some state variables that are common to random
     * forest learners. 
     */
    template <class L>
    class RandomForestLearnerState : public ForestLearnerState {
    public:
        /**
         * Prints the state into the console. 
         */
        virtual void print() const
        {
            ForestLearnerState::print();
            
            for (size_t t = 0; t < treeLearnerStates.size(); t++)
            {
                std::cout << "THREAD " << (t+1) << std::endl;
                treeLearnerStates[t].print();
                std::cout << std::endl;
            }
        }

        /**
         * Resets the state
         */
        virtual void reset()
        {
            for (size_t t = 0; t < treeLearnerStates.size(); t++)
            {
                treeLearnerStates[t].reset();
            }
        }

        /**
         * The states of the individual tree learners (per thread)
         */
        std::vector<typename L::State> treeLearnerStates;
    };
    
    /**
     * This is the basic learner interface
     */
    template <class T>
    class LearnerInterface {
    public:
        using HypothesisType = T;
    };
    
    /**
     * This interface defines the API of offline learners. T is the learned
     * object. 
     */
    template<class T>
    class OfflineLearnerInterface : public LearnerInterface<T> {
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
     * This interface defines the API of online learners. T is the learned
     * object. 
     */
    template<class T>
    class OnlineLearnerInterface : public LearnerInterface<T> {
    public:
        /**
         * Learns a classifier.
         * 
         * @param storage The storage to train the classifier on
         * @return The trained classifier
         */
        virtual std::shared_ptr<T> learn(AbstractDataStorage::ptr storage) = 0;
        
        /**
         * Updates an already learned classifier. 
         * 
         * @param storage The storage to train the classifier on
         * @param classifier The base classifier
         * @return The trained classifier
         */
        virtual std::shared_ptr<T> learn(AbstractDataStorage::ptr storage, std::shared_ptr<T> classifier) = 0;
    };
    
    /**
     * This is an abstract decision tree learning providing functionality
     * needed for all decision tree learners (online or offline).
     */
    class AbstractTreeLearner {
    public:
        AbstractTreeLearner() : 
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
    };
    
    /**
     * This is a an abstract random forest learner providing functionality for
     * online and offline learning.
     */
    class AbstractForestLearner {
    public:
        
        AbstractForestLearner() : numTrees(8), numThreads(1) {}
        
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
        
    protected:
        /**
         * The number of trees that we shall learn
         */
        int numTrees;
        /**
         * The number of threads that shall be used to learn the forest
         */
        int numThreads;
    };
    
    /**
     * This class creates a GUI that outputs the learner's state from time to 
     * time. The template parameter names the learning class. The class has
     * to have a subclass called "State". 
     */
    template <class S>
    class ConsoleGUI {
    public:
        /**
         * @param state The state the GUI should observe
         */
        ConsoleGUI(const S & state) : 
                state(state), 
                interval(1)
        {
            workerThread = std::thread(& ConsoleGUI<S>::worker, this);
        }
        
        /**
         * This function creates the console output. 
         */
        void worker()
        {
            while (!state.terminated)
            {
                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < 80; j++)
                    {
                        std::cout << '-';
                    }
                    std::cout << std::endl;
                }
                
                std::cout << std::endl << std::endl;
                // Did the learner start yet?
                if (!state.started)
                {
                    // Hm, this is odd
                    std::cout << "The learner hasn't started yet." << std::endl;
                    std::cout << "Did you remember to call learn(storage, state) instead of learn(storage)?" << std::endl;
                }
                else
                {
                    std::cout << "Runtime: " << state.getPassedTimeInSeconds() << "s" << std::endl;
                    state.print();
                }
                
                std::this_thread::sleep_for(std::chrono::seconds(interval));
            }
        }
        
        /**
         * Waits for the worker thread to finish
         */
        void join()
        {
            workerThread.join();
            std::cout << "Training completed in " << state.getPassedTimeInSeconds() << "s." << std::endl; 
        }
        
        /**
         * Sets the interval length. 
         * 
         * @param _interval The time between prints in seconds
         */
        void setInterval(int _interval)
        {
            interval = _interval;
        }
        
        /**
         * Returns the interval length. 
         * 
         * @return The inverval length in seconds
         */
        int getInterval() const
        {
            return interval;
        }
        
    private:
        /**
         * The state that is watched
         */
        const S & state;
        /**
         * The worker thread
         */
        std::thread workerThread;
        /**
         * The time interval in which the state is printed
         */
        int interval;
    };
}

#endif
