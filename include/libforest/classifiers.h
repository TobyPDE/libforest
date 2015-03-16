#ifndef LIBF_CLASSIFIERS_H
#define LIBF_CLASSIFIERS_H

/**
 * This file contains the data structures for the classifiers. There are 
 * basically two kinds ot classifiers:
 * 1. Decision trees
 * 2. Random forests
 */

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

#include "tree.h"
#include "io.h"

namespace libf {
    /**
     * The base class for all classifiers. This allows use to use the evaluation
     * tools for both trees and forests. 
     */
    class AbstractClassifier {
    public:
        typedef std::shared_ptr<AbstractClassifier> ptr;
        
        virtual ~AbstractClassifier() {}
        
        /**
         * Assigns an integer class label to some data point
         */
        virtual int classify(const DataPoint & x) const;
        
        /**
         * Classifies an entire data set and uses the integer values. 
         */
        virtual void classify(AbstractDataStorage::ptr, std::vector<int> & results) const;
        
        /**
         * Returns the class posterior probability p(c|x).
         */
        virtual void classLogPosterior(const DataPoint & x, std::vector<float> & probabilities) const = 0;
    };
    
    /**
     * This is the base class for all tree classifier node data classes. 
     */
    class AbstractTreeClassifierNodeData {
    public:
        /**
         * A histogram that represents a distribution over the class labels
         */
        std::vector<float> histogram;
    };
    
    /**
     * This is the base class for all tree classifiers
     */
    template <class Data>
    class AbstractTreeClassifier : public AbstractAxisAlignedSplitTree<Data>, public AbstractClassifier {
    public:
        
        virtual ~AbstractTreeClassifier() {}
        
        /**
         * Only accept template parameters that extend AbstractTreeClassifierNodeData.
         * Note: The double parentheses are needed.
         */
        BOOST_STATIC_ASSERT((boost::is_base_of<AbstractTreeClassifierNodeData, Data>::value));
        
        typedef std::shared_ptr<AbstractTreeClassifier<Data> >  ptr;
        
        /**
         * Returns the class log posterior log(p(c | x)). The probabilities are
         * not normalized. 
         * 
         * @param x The data point for which the log posterior shall be evaluated. 
         * @param probabilities The vector of log posterior probabilities
         */
        void classLogPosterior(const DataPoint & x, std::vector<float> & probabilities) const
        {
            // Get the leaf node
            const int leafNode = this->findLeafNode(x);
            probabilities = this->getNodeData(leafNode).histogram;
        }
    };
    
    /**
     * This is the data each node in a decision tree carries
     */
    class DecisionTreeNodeData : public AbstractTreeClassifierNodeData {};
    
    /**
     * Overload the read binary method to also read DecisionTreeNodeData
     */
    template <>
    inline void readBinary(std::istream & stream, DecisionTreeNodeData & v)
    {
        readBinary(stream, v.histogram);
    }
    
    /**
     * Overload the write binary method to also write DecisionTreeNodeData
     */
    template <>
    inline void writeBinary(std::ostream & stream, const DecisionTreeNodeData & v)
    {
        writeBinary(stream, v.histogram);
    }
    
    /**
     * This class represents a decision tree.
     */
    class DecisionTree : public AbstractTreeClassifier<DecisionTreeNodeData> {
    public:
        typedef std::shared_ptr<DecisionTree> ptr;
    };
    
    /**
     * This is the data each node in an online decision tree carries
     * 
     * These following statistics are saved as entropy histograms.
     * @see http://lrs.icg.tugraz.at/pubs/saffari_olcv_09.pdf
     */
    class OnlineDecisionTreeNodeData : public AbstractTreeClassifierNodeData {
    public:
        /**
         * The node's statistics saved as entropy histogram.
         */
        EfficientEntropyHistogram nodeStatistics;
        /**
         * Left child statistics for all splits.
         */
        std::vector<EfficientEntropyHistogram> leftChildStatistics;
        /**
         * Right child statistics for all splits.
         */
        std::vector<EfficientEntropyHistogram> rightChildStatistics;
        /**
         * Thresholds for each node.
         */
        std::vector< std::vector<float> > nodeThresholds; // TODO: This is really messy!
        /**
         * Features for all nodes.
         */
        std::vector<int> nodeFeatures;
    };
    
    /**
     * Overload the read binary method to also read OnlineDecisionTreeNodeData
     */
    template <>
    inline void readBinary(std::istream & stream, OnlineDecisionTreeNodeData & v)
    {
        // TODO: Implement this stuff
    }
    
    /**
     * Overload the write binary method to also write DecisionTreeNodeData
     */
    template <>
    inline void writeBinary(std::ostream & stream, const OnlineDecisionTreeNodeData & v)
    {
        // TODO: Implement this stuff
    }
    
    /**
     * This class represents a decision tree.
     */
    class OnlineDecisionTree : public AbstractTreeClassifier<OnlineDecisionTreeNodeData> {
    public:
        typedef std::shared_ptr<OnlineDecisionTree> ptr;
    };
    
    /**
     * A random forests that classifies a data point that classifies a data 
     * point using the class posterior estimates of several decision trees. 
     */
    template <class T>
    class AbstractRandomForest : public AbstractClassifier {
    public:
        /**
         * Only accept template parameters that extend AbstractClassifier.
         * Note: The double parentheses are needed.
         */
        BOOST_STATIC_ASSERT((boost::is_base_of<AbstractClassifier, T>::value));
        
        typedef std::shared_ptr<AbstractRandomForest<T> > ptr;
        
        virtual ~AbstractRandomForest() {}
        
        /**
         * Reads the tree from a stream. 
         * 
         * @param stream the stream to read the forest from.
         */
        virtual void read(std::istream & stream)
        {
            // Read the number of trees in this ensemble
            int size;
            readBinary(stream, size);

            // Read the trees
            for (int i = 0; i < size; i++)
            {
                std::shared_ptr<T> tree = std::make_shared<T>();

                tree->read(stream);
                addTree(tree);
            }
        }
        
        /**
         * Writes the tree to a stream
         * 
         * @param stream The stream to write the forest to. 
         */
        void write(std::ostream & stream) const
        {
            // Write the number of trees in this ensemble
            writeBinary(stream, getSize());

            // Write the individual trees
            for (int i = 0; i < getSize(); i++)
            {
                getTree(i)->write(stream);
            }
        }
        
        /**
         * Returns the class log posterior log(p(c | x)).
         * 
         * @param x The data point x to determine the posterior distribution of
         * @param probabilities A vector of log posterior probabilities
         */
        virtual void classLogPosterior(const DataPoint & x, std::vector<float> & probabilities) const
        {
            BOOST_ASSERT_MSG(getSize() > 0, "Cannot classify a point from an empty ensemble.");

            trees[0]->classLogPosterior(x, probabilities);

            // Let the crowd decide
            for (size_t i = 1; i < trees.size(); i++)
            {
                // Get the probabilities from the current tree
                std::vector<float> currentHist;
                trees[i]->classLogPosterior(x, currentHist);

                BOOST_ASSERT(currentHist.size() > 0);

                // Accumulate the votes
                for (size_t c = 0; c < currentHist.size(); c++)
                {
                    probabilities[c] += currentHist[c];
                }
            }
        }
        
        /**
         * Adds a tree to the ensemble
         * 
         * @param tree The tree to add to the ensemble
         */
        void addTree(std::shared_ptr<T> tree)
        {
            trees.push_back(tree);
        }
        
        /**
         * Returns the number of trees
         * 
         * @return The number of trees in this ensemble
         */
        int getSize() const
        {
            return trees.size();
        }
        
        /**
         * Returns the i-th tree
         * 
         * @param i The index of the tree to return
         * @return The i-th tree
         */
        std::shared_ptr<T> getTree(int i) const
        {
            BOOST_ASSERT_MSG(0 <= i && i < getSize(), "Invalid tree index.");
            
            return trees[i];
        }
        
        /**
         * Removes the i-th tree from the ensemble. 
         * 
         * @param i The index of the tree
         */
        void removeTree(int i)
        {
            BOOST_ASSERT_MSG(0 <= i && i < getSize(), "Invalid tree index.");
            
            // Remove it from the array
            trees.erase(trees.begin() + i);
        }
        
    private:
        /**
         * The individual decision trees. 
         */
        std::vector< std::shared_ptr<T> > trees;
    };
    
    /**
     * This is a traditional random forest for offline learning. 
     */
    class RandomForest : public AbstractRandomForest<DecisionTree> 
    {
    public:
        typedef std::shared_ptr<RandomForest> ptr;
        RandomForest() : AbstractRandomForest<DecisionTree> () {}
        virtual ~RandomForest() {}
    };
    
    /**
     * This is an online random forest for online learning. 
     */
    class OnlineRandomForest : public AbstractRandomForest<OnlineDecisionTree> 
    {
    public:
        typedef std::shared_ptr<OnlineRandomForest> ptr;
        OnlineRandomForest() : AbstractRandomForest<OnlineDecisionTree> () {}
        virtual ~OnlineRandomForest() {}
    };
    
    /**
     * A boosted random forest classifier.
     */
    class BoostedRandomForest : public AbstractClassifier {
    public:
        typedef std::shared_ptr<BoostedRandomForest> ptr;
        
        virtual ~BoostedRandomForest() {}
        
        /**
         * Reads the tree from a stream. 
         * 
         * @param stream The stream to read the classifier from. 
         */
        virtual void read(std::istream & stream);
        
        /**
         * Writes the tree to a stream
         * 
         * @param stream The stream to write the classifier to. 
         */
        virtual void write(std::ostream & stream) const;
        
        /**
         * Returns the class log posterior p(c | x). In this case, this cannot
         * be seen as a probability because of the boosting effects.
         * 
         * @param x The data point to calculate the posterior distribution for
         * @param probabilities A vector of log posterior probabilities
         */
        virtual void classLogPosterior(const DataPoint & x, std::vector<float> & probabilities) const;
        
        /**
         * Adds a tree to the ensemble
         * 
         * @param tree The tree to add to the ensemble
         * @param weight The weight of the tree in the classification process
         */
        void addTree(DecisionTree::ptr tree, float weight)
        {
            BOOST_ASSERT_MSG(weight >= 0, "Tree weight must not be negative.");
            
            trees.push_back(tree);
            weights.push_back(weight);
        }
        
        /**
         * Returns the number of trees
         * 
         * @return The total number of trees.
         */
        int getSize() const
        {
            return trees.size();
        }
        
        /**
         * Returns the i-th tree
         * 
         * @param i The index of the tree
         * @return The i-th tree
         */
        DecisionTree::ptr getTree(int i) const
        {
            BOOST_ASSERT_MSG(0 <= i && i < getSize(), "Invalid tree index.");
            
            return trees[i];
        }
        
        /**
         * Removes the i-th tree from the ensemble. 
         * 
         * @param i The index of the tree
         */
        void removeTree(int i)
        {
            BOOST_ASSERT_MSG(0 <= i && i < getSize(), "Invalid tree index.");
            
            // Remove it from the array
            trees.erase(trees.begin() + i);
            weights.erase(weights.begin() + i);
        }
        
    private:
        /**
         * The individual decision trees. 
         */
        std::vector<DecisionTree::ptr> trees;
        /**
         * The tree weights
         */
        std::vector<float> weights;
    };
}
#endif