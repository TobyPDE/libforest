#include "libforest/data_tools.h"

using namespace libf;

////////////////////////////////////////////////////////////////////////////////
/// KMeans
////////////////////////////////////////////////////////////////////////////////

void KMeans::cluster(AbstractDataStorage::ptr storage, 
        AbstractDataStorage::ptr centers, std::vector<int> & labels)
{
    BOOST_ASSERT_MSG(storage->getSize() > 0, "Cannot run k-means on an empty data storage.");
    
    const int N = storage->getSize();
    const int D = storage->getDimensionality();
    const int K = numClusters;
    const int T = numIterations;
    const int M = numTries;
    
    switch(centerInitMethod)
    {
        case CENTERS_RANDOM:
            initCentersRandom(storage, centers);
            break;
        case CENTERS_PP:
            initCentersPP(storage, centers);
            break;
    }
    
    DataStorage::ptr bestCenters = DataStorage::Factory::create();
    std::vector<int> bestLabels(N, 0);
    
    for (int m = 0; m < M; m++)
    {
        // Reference for some notation [1]:
        //  C. Elkan.
        //  Using the Triangle Inequality to Accelerate K-Means.
        //  international COnference on Machine Learning, 2003.
        
        // Initialize all needed data for this k-means run:
        DataStorage::ptr currentCenters = std::shared_ptr<DataStorage>(new DataStorage(*centers));
        std::vector<int> currentLabels(N, 0);
        
        // u(x) in [1]:
        std::vector<float> currentDistances(N, 1e35);
        
        // l(x, c) in [1]:
        Eigen::MatrixXf lowerBound(N, K);
        
        // Distance amtrix between centers.
        Eigen::MatrixXf clusterDistances(K, K);
        
        for (int k = 0; k < K; k++)
        {
            for (int l; l < K; l++)
            {
                DataPoint difference = currentCenters->getDataPoint(k) 
                        - currentCenters->getDataPoint(l);
                
                clusterDistances(k, l) = difference.transpose()*difference;
            }
        }
        
        // Assign each point to the nearest cluster using the triangle inequality.
        for (int n = 0; n < N; n++)
        {
            for (int k = 0; k < K; k++)
            {
                // It holds: if d(c', c) >= 2*d(c, x) then d(x,c) >= d(x, c'):)
//                if (clusterDistances(k, currentLabels[n]) < 2*currentDistances[n])
//                {
                    DataPoint difference = storage->getDataPoint(n)
                            - currentCenters->getDataPoint(k);
                    
                    float distance = difference.transpose()*difference;
                    
                    // Set the lower bound l(x, c):
                    lowerBound(n, k) = distance;
                    
                    if (distance < currentDistances[n])
                    {
                        currentDistances[n] = distance;
                    }
//                }
            }
        }
        
        // r(x) in [1]:
        std::vector<bool> computePointClusterDistance(N, true);
        
        for (int t = 0; t < T; t++)
        {
            // Update the between center distances:
            for (int k = 0; k < K; k++)
            {
                for (int l; l < K; l++)
                {
                    DataPoint difference = currentCenters->getDataPoint(k) 
                            - currentCenters->getDataPoint(l);

                    clusterDistances(k, l) = difference.transpose()*difference;
                }
            }
            
            // s(c) in [1):
            std::vector<float> minClusterDistance(K, 1e35);
            
            // For each center: compute minimum distance to the next center.
            for (int k = 0; k < K; k++)
            {
                for (int l = 0; l < K; l++)
                {
                    if (clusterDistances(k, l) < minClusterDistance[k])
                    {
                        minClusterDistance[k] = clusterDistances(k, l);
                    }
                }
            }
            
            for (int n = 0; n < N; n++)
            {
                // Current cluster of this point.
                const int cluster = currentLabels[n];
                
                // If u(x) <= 1/2 * s(c(x)), then skip this point.
                if (currentDistances[n] <= 1.f/2.f * minClusterDistance[cluster])
                {
                    continue;
                }
                
                for (int k = 0; k < K; k++)
                {
                    if (k == cluster) 
                    {
                        continue;
                    }
                    
                    if (currentDistances[n] <= lowerBound(n, k))
                    {
                        continue;
                    }
                    
                    if (currentDistances[n] <= 1.f/2.f * clusterDistances(cluster, k))
                    {
                        continue;
                    }
                    
                    float distance  = 0;
                    if (computePointClusterDistance[n])
                    {
                        // We need to compute the difference to the current cluster.
                        DataPoint difference = storage->getDataPoint(n)
                                - currentCenters->getDataPoint(cluster);
                        
                        distance = difference.transpose()*difference;
                    }
                    else
                    {
                        distance = currentDistances[n];
                    }
                    
                    if (distance > lowerBound(n, k) 
                            || distance > 1.f/2.f * clusterDistances(clsuter, k))
                    {
                        // Compute difference to new center.
                        DataPoint difference = storage->getDataPoint(n)
                                - currentCenters->getDataPoint(k);
                        
                        float newDistance = difference.transpose()*difference;
                        
                        if (newDistance < distance)
                        {
                            currentLabels[n] = k;
                        }
                    }
                }
            }
            
            // Compute new centers.
            DataStorage::ptr newCenters = DataStorage::Factory::create();
            std::vector<int> newCenterCount(K, 0);
            
            for (int n = 0; n < N; n++)
            {
                const int cluster = currentLabels[n];
                
                newCenters->getDataPoint(cluster) += storage->getDataPoint(n);
                newCenterCount[cluster]++;
            }
            
            // Compute actual means.
            for (int k = 0; k < K; k++)
            {
                newCenters->getDataPoint(k) /= newCenterCount[k];
            }
            
            // Compute the distance of each center to the new version.
            
            for (int n = 0; n < N; n++)
            {
                for (int k = 0; k < K; k++)
                {
                    
                }
            }
        }
    }
}

void KMeans::initCentersPP(AbstractDataStorage::ptr storage, AbstractDataStorage::ptr centers)
{
    const int N = storage->getSize();
    const int D = storage->getDimensionality();
    
    // K-means++ initialization.
    for (int k = 0; k < numClusters; k++)
    {
        // The probability dsitribution over all points we draw from.
        Eigen::VectorXf probability(N);

        if (k == 0)
        {
            // If this is the first center, then probability is unifrom.
            for (int n = 0; n < N; n++)
            {
                probability(n) = 1;
            }
        }
        else
        {
            // Compute distance to nearest center for each data point.
            for (int n = 0; n < N; n++)
            {
                float minDistance = 1e35;
                for (int c = 0; c < k; c++)
                {

                    DataPoint difference = storage->getDataPoint(n) - centers->getDataPoint(k);
                    float distance = difference.transpose()*difference;
                    
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                    }
                }
                
                probability(n) = minDistance;
            }
        }
        
        // Normalize by the sum of all distances.
        probability /= probability.sum();

        // Compute the cumulative sum of all probabilities in order
        // to draw from the distribution.
        std::vector<float> probabilityCumSum(N, 0);
        for (int n = 0; n < N; n++)
        {
            if (n == 0)
            {
                probabilityCumSum[n] = probability(n);
            }
            else
            {
                probabilityCumSum[n] = probability(n) + probabilityCumSum[n - 1];
            }
        }

        // Choose random number in [0,1];
        float r = std::rand()/static_cast<float>(RAND_MAX);

        // Find the corresponding data point index.
        int n = 0;
        while (r > probabilityCumSum[n]) {
            n++;
        }

        // We overstepped the drawn point by one.
        n = std::max(0, n - 1);
        DataPoint center(storage->getDataPoint(n)); // Copy the center!
        centers->addDataPoint(center);
    }
}