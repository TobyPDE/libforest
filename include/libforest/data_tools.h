#ifndef LIBF_DATA_TOOLS_H
#define	LIBF_DATA_TOOLS_H

#include "data.h"

namespace libf {
    
    /**
     * A fast k-means implementation based on the following references:
     * 
     *  D. Arthur, S. Vassilvitskii.
     *  k-means++: The Advantage of Careful Seeding
     *  Proceedings of the ACM-SIAM Symposium on DIscrete Algorithms, 2007.
     * 
     *  C. Elkan.
     *  Using the Triangle Inequality to Accelerate k-Means.
     *  International Conference on Machine Learning, 2003.
     */
    class KMeans {
    public:
        
        typedef std::shared_ptr<KMeans> ptr;
        
        /**
         * Choose centers from a uniform distribution over all data points.
         */
        const static int CENTERS_RANDOM = 1;
        /**
         * Choose centers according to the k-means++ paradigm proportional to
         * their squared distance to all other centers.
         */
        const static int CENTERS_PP = 2;
        
        /**
         * Constructor.
         */
        KMeans() :
                numClusters(2),
                numTries(5), 
                numIterations(50),
                centerInitMethod(KMeans::CENTERS_PP) {};
                
        /**
         * Destructor.
         */
        ~KMeans() {};
        
        /**
         * Sets the number of clusters to generate.
         * 
         * @param _numClusters THe number of clusters to generate
         */
        void setNumClusters(int _numClusters)
        {
            numClusters = _numClusters;
        }
        
        /**
         * Returns the number of clusters to generate.
         * 
         * @return The number of clusters to geenrate
         */
        int getNumClusters()
        {
            return numClusters;
        }
        
        /**
         * Sets the number of retries to run.
         * 
         * @param _numTries number of retries
         */
        void setNumTries(int _numTries)
        {
            numTries = _numTries;
        }
        
        /**
         * Returns the number of retries used.
         * 
         * @return The number of retries used
         */
        int getNumTries()
        {
            return numTries;
        }
        
        /**
         * Sets the maximum number of iterations.
         * 
         * @param _numIterations The maximum number of iterations.
         */
        void setNumIterations(int _numIterations)
        {
            numIterations = _numIterations;
        }
        
        /**
         * Returns the maximum number of iterations.
         * 
         * @return The maximum number of iterations.
         */
        int getNumIterations()
        {
            return numIterations;
        }
        
        /**
         * Sets the method to use to choose the initial center.
         * 
         * @param _centerInitMethod The method to choose the initial centers
         */
        void setCenterInitMethod(int _centerInitMethod)
        {
            switch(_centerInitMethod)
            {
                case CENTERS_RANDOM:
                case CENTERS_PP:
                    centerInitMethod = _centerInitMethod;
                    break;
                default:
                    BOOST_ASSERT_MSG(false, "Unknown center initialization method.");
                    break;
            }
        }
        
        /**
         * Returns the used method to initialize the centers.
         * 
         * @return The method used to initialize the centers
         */
        int getCenterInitMethod()
        {
            return centerInitMethod;
        }
        
        /**
         * Run k-means clustering on the given data storage and write the found
         * centers and labels into the corresponding containers.
         * 
         * @param storage The data storage to run k-means on
         * @param centers The found centers will be written here
         * @param labels The corresponding labels will be written here
         */
        float cluster(AbstractDataStorage::ptr storage, 
                AbstractDataStorage::ptr centers, std::vector<int> & labels);
        
    private:
        /**
         * Initialize the centers according to k-means++.
         * 
         * @param storage The data storage containing all points to cluster
         * @param centers The initial centers are written here
         */
        void initCentersPP(AbstractDataStorage::ptr storage, DataStorage::ptr centers);
        
        /**
         * Initialize the centers randomly
         * 
         * @param storage The data storage containing all points to cluster
         * @param centers The initial centers are written here
         */
        void initCentersRandom(AbstractDataStorage::ptr storage, DataStorage::ptr centers);
        
        /**
         * Number of clusters to generate.
         */
        int numClusters;
        /**
         * Number of retries in order to minimize objective.
         */
        int numTries;
        /**
         * Number of iterations per try.
         */
        int numIterations;
        /**
         * Initialization method to find godd centers.
         */
        int centerInitMethod;
    };
}

#endif	/* LIBF_DATA_TOOLS_H */

