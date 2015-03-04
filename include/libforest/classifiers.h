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
     * A histogram over the class labels. We use this for training.
     */
    class EfficientEntropyHistogram {
    public:
        /**
         * Default constructor
         */
        EfficientEntropyHistogram() : 
                bins(0), 
                histogram(0), 
                mass(0), 
                entropies(0), 
                totalEntropy(0) {}

        /**
         * Construct a entropy histogram of the given size.
         */
        EfficientEntropyHistogram(int _classCount) : 
            bins(_classCount), 
            histogram(0), 
            entropies(0), 
            mass(0), 
            totalEntropy(0) { resize(_classCount); }

        /**
         * Copy constructor
         */
        EfficientEntropyHistogram(const EfficientEntropyHistogram & other) : EfficientEntropyHistogram()
        {
            // Prevent assignment of empty histogram.
            if (other.bins > 0)
            {
                resize(other.bins);
                for (int i = 0; i < bins; i++)
                {
                    set(i, other.at(i));
                }
                mass = other.mass;
            }
        }

        /**
         * Assignment operator
         */
        EfficientEntropyHistogram & operator= (const EfficientEntropyHistogram &other)
        {
            // Prevent self assignment
            if (this != &other)
            {
                if (other.bins != bins)
                {
                    resize (other.bins);
                }
                for (int i = 0; i < bins; i++)
                {
                    set(i, other.at(i));
                    entropies[i] = other.entropies[i];
                }
                mass = other.mass;
                totalEntropy = other.totalEntropy;
            }
            
            return *this;
        }

        /**
         * Destructor
         */
        ~EfficientEntropyHistogram()
        {
            if (histogram != 0)
            {
                delete[] histogram;
            }
            if (entropies != 0)
            {
                delete[] entropies;
            }
        }

        /**
         * Resizes the histogram to a certain size
         */
        void resize(int _classCount)
        {
            // Release the current histogram
            if (histogram != 0)
            {
                delete[] histogram;
                histogram = 0;
            }
            if (entropies != 0)
            {
                delete[] entropies;
                entropies = 0;
            }

            // Only allocate a new histogram, if there is more than one class
            if (_classCount > 0)
            {
                histogram = new int[_classCount];
                entropies = new float[_classCount];
                bins = _classCount;

                // Initialize the histogram
                for (int i = 0; i < bins; i++)
                {
                    histogram[i] = 0;
                    entropies[i] = 0;
                }
            }
        }

        /**
         * Returns the size of the histogram (= class count)
         */
        int size() const { return bins; }

        /**
         * Get the histogram value for class i.
         */
        int at(const int i) const
        {
            assert(i >= 0 && i < bins);
            return histogram[i];
        }
        
        /**
         * Add v instances to class i.
         */
        void add(const int i, const int v)
        {
            assert(i >= 0 && i < bins);
            mass += v; histogram[i] += v;
        }
        
        /**
         * Remove v instances from class i.
         */
        void sub(const int i, const int v)
        {
            assert(i >= 0 && i < bins);
            mass -= v; histogram[i] -= v;
        }
        
        /**
         * Add one instance to class i.
         */
        void add1(const int i)
        {
            assert(i >= 0 && i < bins);
            mass += 1; histogram[i]++;
        }
        
        /**
         * Remove one instance from class i.
         */
        void sub1(const int i)
        {
            assert(i >= 0 && i < bins);
            mass -= 1; histogram[i]--;
        }
        
        /**
         * Add one instance of class i while updating entropy information.
         */
        void addOne(const int i);
        
        /**
         * Remove one instance of class i while updating the entropy information.
         */
        void subOne(const int i);

        /**
         * Returns the mass
         */
        float getMass() const
        {
            return mass;
        }

        /**
         * Calculates the entropy of a histogram
         * 
         * @return The calculated entropy
         */
        float entropy() const
        {
            return totalEntropy;
        }

        /**
         * Initializes all entropies
         */
        void initEntropies();

        /**
         * Sets all entries in the histogram to 0
         */
        void reset()
        {
            for (int i = 0; i < bins; i++)
            {
                histogram[i] = 0;
                entropies[i] = 0;
            }
            totalEntropy = 0;
            mass = 0;
        }

        /**
         * Returns the greatest bin
         */
        int argMax() const
        {
            int maxBin = 0;
            int maxCount = histogram[0];
            for (int i = 1; i < bins; i++)
            {
                if (histogram[i] > maxCount)
                {
                    maxCount = at(i);
                    maxBin = i;
                }
            }

            return maxBin;
        }

        /**
         * Returns true if the histogram is pure
         */
        bool isPure() const
        {
            bool nonPure = false;
            for (int i = 0; i < bins; i++)
            {
                if (histogram[i] > 0)
                {
                    if (nonPure)
                    {
                        return false;
                    }
                    else
                    {
                        nonPure = true; 
                    }
                }
            }
            return true;
        }

    private:
        // int get(const int i) const { return histogram[i]; }
        /**
         * Set the value of class i to v.
         */
        void set(const int i, const int v)
        {
            assert(i >= 0 && i < bins);
            mass -= histogram[i]; mass += v; histogram[i] = v;
        }
        
        /**
         * The number of classes in this histogram
         */
        int bins;

        /**
         * The actual histogram
         */
        int* histogram;

        /**
         * The integral over the entire histogram
         */
        float mass;

        /**
         * The entropies for the single bins
         */
        float* entropies;

        /**
         * The total entropy
         */
        float totalEntropy;

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
         * Creates a decision tree which maintains a set of statistics
         * for each leaf node.
         */
        DecisionTree(bool _statistics);
        
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
        
        /**
         * Get depth of a node.
         */
        int getDepth(int node)
        {
            return depths[node];
        }
        
        /**
         * Get node statistics (i.e. an entropy histogram).
         * 
         */
        EfficientEntropyHistogram & getNodeStatistics(int node)
        {
            return nodeStatistics[node];
        }
        
        /**
         * Get node thresholds (a threshold for each sampled feature).
         */
        std::vector< std::vector<float> > & getNodeThresholds(int node)
        {
            return nodeThresholds[node];
        }
        
        /**
         * Get node features.
         */
        std::vector<int> & getNodeFeatures(int node)
        {
            return nodeFeatures[node];
        }
        
        /**
         * Get left child statistics for each combination of feature and threshold.
         * @param node
         * @return 
         */
        std::vector<EfficientEntropyHistogram> & getLeftChildStatistics(int node)
        {
            return leftChildStatistics[node];
        }
        
        /**
         * Get right child statistics for each combination of feature and threshold.
         * @param node
         * @return 
         */
        std::vector<EfficientEntropyHistogram> & getRightChildStatistics(int node)
        {
            return rightChildStatistics[node];
        }
        
    private:
        /**
         * Adds a plain new node.
         */
        void addNode(int depth);
        
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