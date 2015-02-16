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

    /**
     * This class represents a decision tree.
     */
    class DecisionTree : public Classifier {
    public:
        /**
         * Creates a new decision tree
         */
        DecisionTree();
        
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
        int splitNode(int node);
        
        /**
         * Returns the leaf node for a specific data point
         */
        int findLeafNode(const DataPoint* x) const;
        
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
            return histograms[node];
        }
        
        /**
         * Returns the histogram for a node
         */
        const std::vector<float> & getHistogram(int node) const
        {
            return histograms[node];
        }
        
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
            return leftChild[node] == 0;
        }
        
        /**
         * Returns the left child for a node
         */
        int getLeftChild(int node) const
        {
            return leftChild[node];
        }
        
    private:
        /**
         * Adds a plain new node.
         */
        void addNode();
        
        /**
         * The split feature at each node. 
         */
        std::vector<int> splitFeatures;
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
         * The histograms for each node. We only store actual histograms at 
         * leaf nodes. 
         */
        std::vector< std::vector<float> > histograms;
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
        void addTree(Classifier* tree)
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
        Classifier* getTree(int i)
        {
            return trees[i];
        }
        
        /**
         * Returns the i-th tree
         */
        const Classifier* getTree(int i) const
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
        std::vector<Classifier*> trees;
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
        void addTree(Classifier* tree, float weight)
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
        Classifier* getTree(int i)
        {
            return trees[i];
        }
        
        /**
         * Returns the i-th tree
         */
        const Classifier* getTree(int i) const
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
        std::vector<Classifier*> trees;
        /**
         * The tree weights
         */
        std::vector<float> weights;
    };
}
#endif