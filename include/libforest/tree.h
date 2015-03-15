#ifndef LIBF_TREE_H
#define LIBF_TREE_H

#include <vector>

#include "util.h"
#include "error_handling.h"
#include "io.h"
#include "data.h"

namespace libf {
    
    /**
     * This is a base class for trees that split the space using axis aligned
     * splits. Each node in the tree can carry some specified data. 
     */
    template <class Data>
    class SplitTree {
    public:
        /**
         * Creates an empty split tree.
         */
        SplitTree()
        {
            // Reserve some memory for the nodes
            // This speeds up training a bit
            splitFeatures.reserve(LIBF_GRAPH_BUFFER_SIZE);
            thresholds.reserve(LIBF_GRAPH_BUFFER_SIZE);
            leftChild.reserve(LIBF_GRAPH_BUFFER_SIZE);
            depths.reserve(LIBF_GRAPH_BUFFER_SIZE);
            nodeData.reserve(LIBF_GRAPH_BUFFER_SIZE);
        }
        
        /**
         * Destructor.
         */
        virtual ~SplitTree() {}
        
        /**
         * Splits a child node and returns the index of the left child. 
         * 
         * @param node The node index
         * @return The index of the left child node
         */
        int splitNode(int node)
        {
            // Make sure this is a valid node ID
            BOOST_ASSERT_MSG(0 <= node && node < static_cast<int>(splitFeatures.size()), "Invalid node index.");
            // Make sure this is a former leaf node
            BOOST_ASSERT_MSG(leftChild[node] == 0, "Cannot split non-leaf node.");

            // Determine the index of the new left child
            const int leftNode = static_cast<int>(splitFeatures.size());

            // Add the child nodes
            const int depth = depths[node] + 1;
            addNode(depth);
            addNode(depth);

            // Set the child relation
            leftChild[node] = leftNode;

            return leftNode;
        }
        
        /**
         * Passes the data point through the tree and returns the index of the
         * leaf node it ends up in. 
         * 
         * @param x The data point to pass down the tree
         * @return The index of the leaf node v ends up in
         */
        int findLeafNode(const DataPoint & x) const
        {
            // Select the root node as current node
            int node = 0;

            // Follow the tree until we hit a leaf node
            while (leftChild[node] != 0)
            {
                BOOST_ASSERT(splitFeatures[node] >= 0 && splitFeatures[node] < x.rows());

                // Check the threshold
                if (x(splitFeatures[node]) < thresholds[node])
                {
                    // Go to the left
                    node = leftChild[node];
                }
                else
                {
                    // Go to the right
                    node = leftChild[node] + 1;
                }
            }

            return node;
        }
        
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
        virtual void read(std::istream & stream)
        {
            readBinary(stream, splitFeatures);
            readBinary(stream, thresholds);
            readBinary(stream, leftChild);
            readBinary(stream, depths);
            readBinary(stream, nodeData);
        }
        
        /**
         * Writes the tree to a stream
         * 
         * @param stream The stream to write the tree to.
         */
        virtual void write(std::ostream & stream) const
        {
            writeBinary(stream, splitFeatures);
            writeBinary(stream, thresholds);
            writeBinary(stream, leftChild);
            writeBinary(stream, depths);
            writeBinary(stream, nodeData);
        }
        
        /**
         * Returns the node data. 
         * 
         * @param node The node index
         * @return A reference to the node data
         */
        const Data & getNodeData(int node) const
        {
            BOOST_ASSERT_MSG(0 <= node && node < splitFeatures.size(), "Invalid node index.");
            return nodeData[node];
        }
        
        /**
         * Returns the node data. 
         * 
         * @param node The node index
         * @return A reference to the node data
         */
        Data & getNodeData(int node)
        {
            BOOST_ASSERT_MSG(0 <= node && node < splitFeatures.size(), "Invalid node index.");
            return nodeData[node];
        }
        
        /**
         * Adds a new node. This method needs to be implemented by all trees.
         */
        virtual void addNode(int depth)
        {
            splitFeatures.push_back(0);
            thresholds.push_back(0);
            leftChild.push_back(0);
            depths.push_back(depth);
            nodeData.push_back(Data());
        }
        
    protected:
        
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
         * This is the data each node carries
         */
        std::vector<Data> nodeData;
    };
}

#endif