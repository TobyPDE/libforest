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

#include "data.h"
#include "util.h"
#include "error_handling.h"

namespace libf {
    /**
     * The base class for all classifiers. This allows use to use the evaluation
     * tools for both trees and forests. 
     */
    class Classifier {
    public:
        typedef std::shared_ptr<Classifier> ptr;
        
        virtual ~Classifier() {}
        
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
    
    class Tree {
    public:
        /**
         * Creates an empty density tree.
         */
        Tree();
        
        /**
         * Destructor.
         */
        virtual ~Tree() {};
        
        /**
         * Splits a child node and returns the index of the left child. 
         * 
         * @param node The node index
         */
        int splitNode(int node);
        
        /**
         * Passes the data point through the tree and returns the index of the
         * leaf node it ends up in. 
         * 
         * @param v The data point to pass down the tree
         * @return The index of the leaf node v ends up in
         */
        int findLeafNode(const DataPoint & v) const;
        
        /**
         * Sets the split feature for a node
         * 
         * @param node The index of the node that shall be edited
         * @param feature The feature dimension
         */
        void setSplitFeature(int node, int feature)
        {
            BOOST_ASSERT_MSG(0 <= node && node < splitFeatures.size(), "Invalid node index.");
            BOOST_ASSERT_MSG(feature >= 0, "Invalid feature dimension.");
            
            splitFeatures[node] = feature;
        }
        
        /**
         * Returns the split feature for a node
         * 
         * @param node The index of the node
         * @return The split feature
         */
        int getSplitFeature(int node) const
        {
            BOOST_ASSERT_MSG(0 <= node && node < splitFeatures.size(), "Invalid node index.");
            
            return splitFeatures[node];
        }
        
        /**
         * Sets the threshold for a node
         * 
         * @param node The index of the node
         * @param threshold The new threshold value
         */
        void setThreshold(int node, float threshold)
        {
            BOOST_ASSERT_MSG(0 <= node && node < splitFeatures.size(), "Invalid node index.");
            
            thresholds[node] = threshold;
        }
        
        /**
         * Returns the threshold for a node
         * 
         * @param node The index of the node
         * @return The threshold 
         */
        float getThreshold(int node) const
        {
            BOOST_ASSERT_MSG(0 <= node && node < splitFeatures.size(), "Invalid node index.");
            
            return thresholds[node];
        }
        
        /**
         * Returns the total number of nodes. 
         * 
         * @return The total number of nodes
         */
        int getNumNodes() const
        {
            return static_cast<int>(leftChild.size());
        }
        
        /**
         * Returns true if the given node is a leaf node. 
         * 
         * @param node The index of the node
         * @return True if the node is a leaf node
         */
        bool isLeafNode(int node) const 
        {
            BOOST_ASSERT_MSG(0 <= node && node < splitFeatures.size(), "Invalid node index.");
            
            return leftChild[node] == 0;
        }
        
        /**
         * Returns the index of the left child node for a node. 
         * 
         * @param node The index of the node
         * @return The index of the left child node
         */
        int getLeftChild(int node) const
        {
            BOOST_ASSERT_MSG(0 <= node && node < splitFeatures.size(), "Invalid node index.");
            
            return leftChild[node];
        }
        
        /**
         * Get depth of a node where the root node has depth 0. 
         * 
         * @param node The index of the node
         * @return The depth of the node
         */
        int getDepth(int node)
        {
            BOOST_ASSERT_MSG(0 <= node && node < splitFeatures.size(), "Invalid node index.");
            
            return depths[node];
        }
        
        /**
         * Reads the tree from a stream. 
         * 
         * @param stream The stream to read the tree from
         */
        virtual void read(std::istream & stream);
        
        /**
         * Writes the tree to a stream
         * 
         * @param stream The stream to write the tree to.
         */
        virtual void write(std::ostream & stream) const;
        
        /**
         * Adds a new node. This method needs to be implemented by all trees.
         */
        virtual void addNode(int depth);
        
    protected:
        
        /**
         * Adds a new node. THis method needs to be implemented by all trees.
         */
        virtual void addNodeDerived(int depth) = 0;
        
        /**
         * The depth of each node.
         */
        std::vector<int> depths;
        /**
         * The split feature at each node. 
         */
        std::vector<int> splitFeatures;
        /**
         * The threshold at each node
         */
        std::vector<float> thresholds;
        /**
         * The left child node of each node. If the left child node is 0, then 
         * this is a leaf node. The right child node is left + 1. 
         */
        std::vector<int> leftChild;
        
    };
    
    /**
     * This class represents a decision tree.
     */
    class DecisionTree : public Tree, public Classifier {
    public:
        typedef std::shared_ptr<DecisionTree> ptr;
        
        /**
         * Creates a new decision tree
         */
        DecisionTree();
        
        /**
         * Creates a decision tree which maintains a set of statistics
         * for each leaf node.
         * 
         * @param statistics If true, a set of statistics is maintained for each leaf node
         */
        DecisionTree(bool statistics);
        
        /**
         * Destructor.
         */
        ~DecisionTree() {};
        
        /**
         * Returns the class log posterior log(p(c | x)). The probabilities are
         * not normalized. 
         * 
         * @param x The data point for which the log posterior shall be evaluated. 
         * @param probabilities The vector of log posterior probabilities
         */
        virtual void classLogPosterior(const DataPoint & c, std::vector<float> & probabilities) const;
        
        /**
         * Reads the tree from a stream. 
         * 
         * @param stream The stream to read the tree from
         */
        virtual void read(std::istream & stream);
        
        /**
         * Writes the tree to a stream
         * 
         * @param stream The stream to write the tree to.
         */
        virtual void write(std::ostream & stream) const;
        
        /**
         * Returns a reference the histogram for a node
         * 
         * @param node The index of the node
         * @return A reference to its histogram
         */
        std::vector<float> & getHistogram(int node)
        {
            BOOST_ASSERT_MSG(0 <= node && node < splitFeatures.size(), "Invalid node index.");
            
            return histograms[node];
        }
        
        /**
         * Returns a reference the histogram for a node
         * 
         * @param node The index of the node
         * @return A reference to its histogram
         */
        const std::vector<float> & getHistogram(int node) const
        {
            BOOST_ASSERT_MSG(0 <= node && node < splitFeatures.size(), "Invalid node index.");
            
            return histograms[node];
        }
        
        /**
         * Get node statistics (i.e. an entropy histogram).
         * 
         * @param node The node index
         * @return A reference to the statistics
         */
        EfficientEntropyHistogram & getNodeStatistics(int node)
        {
            BOOST_ASSERT_MSG(0 <= node && node < splitFeatures.size(), "Invalid node index.");
            
            return nodeStatistics[node];
        }
        
        /**
         * Get node thresholds (a threshold for each sampled feature).
         * 
         * @param node The node index
         * @return A reference to the thresholds
         */
        std::vector< std::vector<float> > & getNodeThresholds(int node)
        {
            BOOST_ASSERT_MSG(0 <= node && node < splitFeatures.size(), "Invalid node index.");
            
            return nodeThresholds[node];
        }
        
        /**
         * Get node features.
         * 
         * @param node The index of the node
         * @return A reference to its features
         */
        std::vector<int> & getNodeFeatures(int node)
        {
            BOOST_ASSERT_MSG(0 <= node && node < splitFeatures.size(), "Invalid node index.");
            
            return nodeFeatures[node];
        }
        
        /**
         * Get left child statistics for each combination of feature and threshold.
         * 
         * @param node The node index
         * @return A reference to the left child statistics
         */
        std::vector<EfficientEntropyHistogram> & getLeftChildStatistics(int node)
        {
            return leftChildStatistics[node];
        }
        
        /**
         * Get right child statistics for each combination of feature and threshold.
         * 
         * @param node The index of the node
         * @return A reference to the right child statistics
         */
        std::vector<EfficientEntropyHistogram> & getRightChildStatistics(int node)
        {
            return rightChildStatistics[node];
        }
        
    protected:
        /**
         * Adds a plain new node.
         */
        virtual void addNodeDerived(int depth);
        
    private:
        /**
         * The histograms for each node. We only store actual histograms at 
         * leaf nodes. 
         */
        std::vector< std::vector<float> > histograms;
        /**
         * NOTE ON THE FOLLOWING ATTRIBUTES: It is not clear whether these 
         * statistics (mainly used for online learning) should reside in the 
         * decision tree or the learner. We decided to put them in the decision
         * tree such that a model can be updated/learned independent of the
         * learner.
         * 
         * These statistics are saved as entropy histograms.
         * 
         * @see http://lrs.icg.tugraz.at/pubs/saffari_olcv_09.pdf
         */
        /**
         * Whether to save statistics or not.
         */
        bool statistics;
        /**
         * The node's statistics saved as entropy histogram.
         */
        std::vector<EfficientEntropyHistogram> nodeStatistics;
        /**
         * Left child statistics for all splits.
         */
        std::vector< std::vector<EfficientEntropyHistogram> > leftChildStatistics;
        /**
         * Right child statistics for all splits.
         */
        std::vector< std::vector<EfficientEntropyHistogram> > rightChildStatistics;
        /**
         * Thresholds for each node.
         */
        std::vector< std::vector< std::vector<float> > > nodeThresholds; // TODO: This is really messy!
        /**
         * Features for all nodes.
         */
        std::vector< std::vector<int> > nodeFeatures;
    };
    
    
    /**
     * A random forests that classifies a data point that classifies a data 
     * point using the class posterior estimates of several decision trees. 
     */
    class RandomForest : public Classifier {
    public:
        typedef std::shared_ptr<RandomForest> ptr;
        
        virtual ~RandomForest() {}
        
        /**
         * Reads the tree from a stream. 
         * 
         * @param stream the stream to read the forest from.
         */
        virtual void read(std::istream & stream);
        
        /**
         * Writes the tree to a stream
         * 
         * @param stream The stream to write the forest to. 
         */
        virtual void write(std::ostream & stream) const;
        
        /**
         * Returns the class log posterior log(p(c | x)).
         * 
         * @param x The data point x to determine the posterior distribution of
         * @param probabilities A vector of log posterior probabilities
         */
        virtual void classLogPosterior(const DataPoint & x, std::vector<float> & probabilities) const;
        
        /**
         * Adds a tree to the ensemble
         * 
         * @param tree The tree to add to the ensemble
         */
        void addTree( DecisionTree::ptr tree)
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
        }
        
    private:
        /**
         * The individual decision trees. 
         */
        std::vector<DecisionTree::ptr> trees;
    };
    
    /**
     * A boosted random forest classifier.
     */
    class BoostedRandomForest : public Classifier {
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