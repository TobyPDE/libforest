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

#include "data.h"
#include "util.h"

namespace libf {
    /**
     * Forward declarations to speed up compiling
     */
    class DataStorage;
    class DataPoint;
    
    /**
     * The base class for all classifiers. This allows use to use the evaluation
     * tools for both trees and forests. 
     */
    class Classifier {
    public:
        virtual ~Classifier() {}
        
        /**
         * Assigns an integer class label to some data point
         */
        virtual int classify(const DataPoint* x) const;
        
        /**
         * Classifies an entire data set and uses the integer values. 
         */
        virtual void classify(const DataStorage* storage, std::vector<int> & results) const;
        
        /**
         * Returns the class posterior probability p(c|x).
         */
        virtual void classLogPosterior(const DataPoint* x, std::vector<float> & probabilities) const = 0;
        
        /**
         * Reads the classifier from a stream
         */
        virtual void read(std::istream & stream) = 0;
        
        /**
         * Writes the classifier to a stream
         */
        virtual void write(std::ostream & stream) const = 0;
    };
    
    class Tree {
    public:
        /**
         * Creates a new decision tree
         */
        Tree();
        
        /**
         * Destructor.
         */
        virtual ~Tree() {};
        
        /**
         * Sets the split feature for a node
         */
        void setSplitFeature(int node, int feature)
        {
            splitFeatures[node] = feature;
        }
        
        /**
         * Returns the split feature for a node
         */
        int getSplitFeature(int node) const
        {
            return splitFeatures[node];
        }
        
        /**
         * Sets the threshold for a node
         */
        void setThreshold(int node, float threshold)
        {
            thresholds[node] = threshold;
        }
        
        /**
         * Returns the threshold for a node
         */
        float getThreshold(int node) const
        {
            return thresholds[node];
        }
        
        /**
         * Splits a child node and returns the index of the left child. 
         */
        virtual int splitNode(int node);
        
        /**
         * Returns the leaf node for a specific data point
         */
        int findLeafNode(const DataPoint* x) const;
        
        /**
         * Reads the tree from a stream
         */
        virtual void read(std::istream & stream);
        
        /**
         * Writes the tree to a stream
         */
        virtual void write(std::ostream & stream) const;
        
        /**
         * Returns the total number of nodes
         */
        int getNumNodes() const
        {
            return static_cast<int>(leftChild.size());
        }
        
        /**
         * Returns true if the given node is a leaf node
         */
        bool isLeafNode(int node) const 
        {
            assert(node >= 0 && node <= static_cast<int>(leftChild.size()));
            return leftChild[node] == 0;
        }
        
        /**
         * Returns the left child for a node
         */
        int getLeftChild(int node) const
        {
            assert(node >= 0 && node <= static_cast<int>(leftChild.size()));
            return leftChild[node];
        }
        
        /**
         * Get depth of a node.
         */
        int getDepth(int node)
        {
            assert(node >= 0 && node <= static_cast<int>(depths.size()));
            return depths[node];
        }
        
    protected:
        /**
         * Adds a plain new node.
         */
        virtual void addNode(int depth);
        
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
        using Tree::splitNode;
        
        /**
         * Creates a new decision tree
         */
        DecisionTree();
        
        /**
         * Creates a decision tree which maintains a set of statistics
         * for each leaf node.
         */
        DecisionTree(bool _statistics);
        
        /**
         * Destructor.
         */
        ~DecisionTree() {};
        
        /**
         * Returns the class log posterior p(c | x).
         */
        virtual void classLogPosterior(const DataPoint* x, std::vector<float> & probabilities) const;
        
        /**
         * Reads the tree from a stream
         */
        virtual void read(std::istream & stream);
        
        /**
         * Writes the tree to a stream
         */
        virtual void write(std::ostream & stream) const;
        
        /**
         * Returns the histogram for a node
         */
        std::vector<float> & getHistogram(int node)
        {
            assert(node >= 0 && node <= static_cast<int>(histograms.size()));
            return histograms[node];
        }
        
        /**
         * Returns the histogram for a node
         */
        const std::vector<float> & getHistogram(int node) const
        {
            assert(node >= 0 && node <= static_cast<int>(histograms.size()));
            return histograms[node];
        }
        
        /**
         * Get node statistics (i.e. an entropy histogram).
         * 
         */
        EfficientEntropyHistogram & getNodeStatistics(int node)
        {
            assert(node >= 0 && node <= static_cast<int>(nodeStatistics.size()));
            return nodeStatistics[node];
        }
        
        /**
         * Get node thresholds (a threshold for each sampled feature).
         */
        std::vector< std::vector<float> > & getNodeThresholds(int node)
        {
            assert(node >= 0 && node <= static_cast<int>(nodeThresholds.size()));
            return nodeThresholds[node];
        }
        
        /**
         * Get node features.
         */
        std::vector<int> & getNodeFeatures(int node)
        {
            assert(node >= 0 && node <= static_cast<int>(nodeFeatures.size()));
            return nodeFeatures[node];
        }
        
        /**
         * Get left child statistics for each combination of feature and threshold.
         * @param node
         * @return 
         */
        std::vector<EfficientEntropyHistogram> & getLeftChildStatistics(int node)
        {
            assert(node >= 0 && node <= static_cast<int>(leftChildStatistics.size()));
            return leftChildStatistics[node];
        }
        
        /**
         * Get right child statistics for each combination of feature and threshold.
         * @param node
         * @return 
         */
        std::vector<EfficientEntropyHistogram> & getRightChildStatistics(int node)
        {
            assert(node >= 0 && node <= static_cast<int>(rightChildStatistics.size()));
            return rightChildStatistics[node];
        }
        
    private:
        /**
         * Adds a plain new node.
         */
        virtual void addNode(int depth);
        
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
     * A kernel decision tree. Each split is based on thresholding a kernel 
     * function k(x, x_i) <= t where x_i is some data point from the training
     * set and x is the evaluation point.
     * 
     * TODO: Implement Kernel Trees
     */
    class KernelDecisionTree : public Classifier {
    private:
        /**
         * The data points that are used at every split. 
         */
        std::vector<DataPoint> splitPoints;
        /**
         * The threshold at each node
         */
        std::vector<float> thresholds;
        /**
         * The left child node of each node. If the left child not is 0, then 
         * this is a leaf node. The right child node is left + 1. 
         */
        std::vector<int> leftChild;
        /**
         * The class label for each node
         */
        std::vector<int> classLabels;
        /**
         * The histograms for each node. We only store actual histograms at 
         * leaf nodes. 
         */
        std::vector< std::vector<int> > histograms;
    };
    
    /**
     * Decision DAG class. 
     * TODO: Implement decision DAGs
     */
    class DecisionDAG : public Classifier {
    private:
        /**
         * The split features
         */
        std::vector<int> splitFeatures;
        /**
         * The thresholds
         */
        std::vector<float> thresholds;
        /**
         * The left child relation
         */
        std::vector<int> leftChild;
        /**
         * The right child relation
         */
        std::vector<int> rightChild;
        /**
         * The class label of each node
         */
        std::vector<int> classLabels;
    };
    
    /**
     * A random forests that classifies a data point using a simple majority 
     * voting schema. 
     */
    class RandomForest : public Classifier {
    public:
        virtual ~RandomForest();
        
        /**
         * Reads the tree from a stream
         */
        virtual void read(std::istream & stream);
        
        /**
         * Writes the tree to a stream
         */
        virtual void write(std::ostream & stream) const;
        
        /**
         * Returns the class log posterior p(c | x).
         */
        virtual void classLogPosterior(const DataPoint* x, std::vector<float> & probabilities) const;
        
        /**
         * Adds a tree to the ensemble
         */
        void addTree(DecisionTree* tree)
        {
            trees.push_back(tree);
        }
        
        /**
         * Returns the number of trees
         */
        int getSize() const
        {
            return static_cast<int>(trees.size());
        }
        
        /**
         * Returns the i-th tree
         */
        DecisionTree* getTree(int i)
        {
            return trees[i];
        }
        
        /**
         * Returns the i-th tree
         */
        const DecisionTree* getTree(int i) const
        {
            return trees[i];
        }
        
        /**
         * Removes the i-th tree
         */
        void removeTree(int i)
        {
            // Delete the tree
            delete trees[i];
            // Remove it from the array
            trees.erase(trees.begin() + i);
        }
        
    private:
        /**
         * The individual decision trees. 
         */
        std::vector<DecisionTree*> trees;
    };
    
    /**
     * A boosted random forest classifier.
     */
    class BoostedRandomForest : public Classifier {
    public:
        virtual ~BoostedRandomForest();
        
        /**
         * Reads the tree from a stream
         */
        virtual void read(std::istream & stream);
        
        /**
         * Writes the tree to a stream
         */
        virtual void write(std::ostream & stream) const;
        
        /**
         * Returns the class log posterior p(c | x). In this case, this cannot
         * be seen as a probability because of the boosting effects.
         */
        virtual void classLogPosterior(const DataPoint* x, std::vector<float> & probabilities) const;
        
        /**
         * Adds a tree to the ensemble
         */
        void addTree(DecisionTree* tree, float weight)
        {
            trees.push_back(tree);
            weights.push_back(weight);
        }
        
        /**
         * Returns the number of trees
         */
        int getSize() const
        {
            return static_cast<int>(trees.size());
        }
        
        /**
         * Returns the i-th tree
         */
        DecisionTree* getTree(int i)
        {
            return trees[i];
        }
        
        /**
         * Returns the i-th tree
         */
        const DecisionTree* getTree(int i) const
        {
            return trees[i];
        }
        
        /**
         * Removes the i-th tree
         */
        void removeTree(int i)
        {
            // Delete the tree
            delete trees[i];
            // Remove it from the array
            trees.erase(trees.begin() + i);
            weights.erase(weights.begin() + i);
        }
        
    private:
        /**
         * The individual decision trees. 
         */
        std::vector<DecisionTree*> trees;
        /**
         * The tree weights
         */
        std::vector<float> weights;
    };
}
#endif