#ifndef MCMCF_CLASSIFIERS_H
#define MCMCF_CLASSIFIERS_H

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
        /**
         * Assigns an integer class label to some data point
         */
        virtual int classify(DataPoint* x) const = 0;
        
        /**
         * Classifies an entire data set and uses the integer values. 
         */
        virtual void classify(DataStorage* storage, std::vector<int> & results) const;
        
        /**
         * Assigns a class label to some data point
         */
        //virtual std::string classify(DataPoint* x) const;
        
        /**
         * Classifies an entire data set. 
         */
        //virtual void classify(DataStorage* storage, std::vector<std::string> & results) const;
        
        /**
         * Outputs the class probabilities for a given data point.
         */
        //virtual void classProbabilities(DataPoint* x, std::vector<float> & probabilities) const = 0;
        
        /**
         * Reads the classifier from a stream
         */
        virtual void read(std::istream & stream) = 0;
        
        /**
         * Writes the classifier to a stream
         */
        virtual void write(std::ostream & stream) const = 0;
        
        /**
         * Sets the class label map
         */
        void setClassLabelMap(const std::vector<std::string> & _classLabelMap) 
        {
            classLabelMap = _classLabelMap;
        }
        
        /**
         * Returns the class label map
         */
        const std::vector<std::string> & getClassLabelMap() const
        {
            return classLabelMap;
        }
        
    protected:
        /**
         * The class label map: int -> string
         */
        std::vector<std::string> classLabelMap;
    };
    
    /**
     * This is the abstract parent class of decision trees and decision DAGs
     */
    class GraphClassifier : public Classifier {
    public:
        /**
         * Returns the leaf node for a specific data point
         */
        int findLeafNode(DataPoint* x) const;
        
        /**
         * Returns the histogram for a node
         */
        std::vector<int> & getHistogram(int node)
        {
            return histograms[node];
        }
        
        /**
         * Returns the histogram for a node
         */
        const std::vector<int> & getHistogram(int node) const
        {
            return histograms[node];
        }
        
        /**
         * Returns the mass of a histogram
         */
        int getHistogramMass(int node) const;
        
        /**
         * Calculates the masses of the histograms
         */
        void calcHistogramMasses();
        
    protected:
        /**
         * The histograms for each node. We only store actual histograms at 
         * leaf nodes. 
         */
        std::vector< std::vector<int> > histograms;   
        /**
         * The masses of the histograms
         */
        std::vector<int> histogramMasses;
    };
    
    /**
     * This class represents a decision tree.
     * 
     * TODO: Make this a derived class of GraphClassifier
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
         * Sets the threshold for a node
         */
        void setThreshold(int node, float threshold)
        {
            thresholds[node] = threshold;
        }
        
        /**
         * Sets the class label for a node
         */
        void setClassLabel(int node, int label)
        {
            classLabels[node] = label;
        }
        
        /**
         * Splits a child node and returns the index of the left child. 
         */
        int splitNode(int node);
        
        /**
         * Returns the leaf node for a specific data point
         */
        int findLeafNode(DataPoint* x) const;
        
        /**
         * Classifies a data point
         */
        virtual int classify(DataPoint* x) const;
        
        /**
         * Classifies an entire data set. 
         */
        virtual void classify(DataStorage* storage, std::vector<int> & results) const
        {
            Classifier::classify(storage, results);
        }
        
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
        std::vector<int> & getHistogram(int node)
        {
            return histograms[node];
        }
        
        /**
         * Returns the histogram for a node
         */
        const std::vector<int> & getHistogram(int node) const
        {
            return histograms[node];
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
     * A kernel decision tree. Each split is based on thresholding a kernel 
     * function k(x, x_i) <= t where x_i is some data point from the training
     * set and x is the evaluation point.
     */
    class KernelDecisionTree : public GraphClassifier {
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
     */
    class DecisionDAG : public GraphClassifier {
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
     * 
     * TODO: Use abstract GraphClassifier class
     */
    class RandomForest : public Classifier {
    public:
        RandomForest() : smoothing(1) {}
        
        /**
         * Classifies a data point
         */
        virtual int classify(DataPoint* x) const;
        
        /**
         * Classifies an entire data set. 
         */
        virtual void classify(DataStorage* storage, std::vector<int> & results) const
        {
            Classifier::classify(storage, results);
        }
        
        /**
         * Reads the tree from a stream
         */
        virtual void read(std::istream & stream);
        
        /**
         * Writes the tree to a stream
         */
        virtual void write(std::ostream & stream) const;
        
        /**
         * Adds a tree to the ensemble
         */
        void addTree(GraphClassifier* tree)
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
        GraphClassifier* getTree(int i)
        {
            return trees[i];
        }
        
        /**
         * Returns the i-th tree
         */
        const GraphClassifier* getTree(int i) const
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
        
        /**
         * Sets the smoothing parameter
         */
        void setSmoothing(float _smoothing)
        {
            smoothing = _smoothing;
        }
        
        /**
         * Returns the smoothing parameter
         */
        float getSmoothing() const
        {
            return smoothing;
        }
    private:
        /**
         * The individual decision trees. 
         */
        std::vector<GraphClassifier*> trees;
        /**
         * The smoothing parameter for classification
         */
        float smoothing;
    };
}
#endif