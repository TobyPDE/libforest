#include "libforest/data_tools.h"

using namespace libf;

////////////////////////////////////////////////////////////////////////////////
/// KMeans
////////////////////////////////////////////////////////////////////////////////

float KMeans::cluster(AbstractDataStorage::ptr storage, 
        AbstractDataStorage::ptr centers, std::vector<int> & labels)
{
    BOOST_ASSERT_MSG(storage->getSize() > 0, "Cannot run k-means on an empty data storage.");
    
    const int N = storage->getSize();
    const int D = storage->getDimensionality();
    const int K = numClusters;
    const int T = numIterations;
    const int M = numTries;
    
    AbstractDataStorage::ptr bestCenters = DataStorage::Factory::create();
    std::vector<int> bestLabels(N, 0);
    float bestError = 1e35;
    
    for (int m = 0; m < M; m++)
    {
        DataStorage::ptr initCenters = DataStorage::Factory::create();
        switch(centerInitMethod)
        {
            case CENTERS_RANDOM:
                initCentersRandom(storage, initCenters);
                break;
            case CENTERS_PP:
                initCentersPP(storage, initCenters);
                break;
        }

        BOOST_ASSERT_MSG(initCenters->getSize() == K, "Could not initialize the correct number of centers.");
        
        // Reference for some notation [1]:
        //  C. Elkan.
        //  Using the Triangle Inequality to Accelerate K-Means.
        //  international COnference on Machine Learning, 2003.
        
        // Initialize all needed data for this k-means run:
        AbstractDataStorage::ptr currentCenters = initCenters->copy();
        std::vector<int> currentLabels(N, 0);
        
        // u(x) in [1]:
        std::vector<float> currentDistances(N, 1e35);
        
        // l(x, c) in [1]:
        Eigen::MatrixXf lowerBound(N, K);
        
        // Distance amtrix between centers.
        Eigen::MatrixXf clusterDistances(K, K);
        
        for (int k = 0; k < K; k++)
        {
            for (int l = 0; l < K; l++)
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
                for (int l = 0; l < K; l++)
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
//                    if (k == cluster) 
//                    {
//                        continue;
//                    }
//                    
//                    if (currentDistances[n] <= lowerBound(n, k))
//                    {
//                        continue;
//                    }
//                    
//                    if (currentDistances[n] <= 1.f/2.f * clusterDistances(cluster, k))
//                    {
//                        continue;
//                    }
                    
                    float distance  = 0;
                    if (computePointClusterDistance[n])
                    {
                        // We need to compute the difference to the current cluster.
                        DataPoint difference = storage->getDataPoint(n)
                                - currentCenters->getDataPoint(cluster);
                        
                        distance = difference.transpose()*difference;
                        
//                        computePointClusterDistance[n] = false;
                    }
                    else
                    {
                        distance = currentDistances[n];
                    }
                    
                    if (distance > lowerBound(n, k) 
                            || distance > 1.f/2.f * clusterDistances(cluster, k))
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
            
            // Initialize data storage.
            for (int k = 0; k < K; k++)
            {
                newCenters->addDataPoint(Eigen::VectorXf::Zero(D));
            }
            
            // Compute center means.
            for (int n = 0; n < N; n++)
            {
                const int cluster = currentLabels[n];
                
                newCenters->getDataPoint(cluster) += storage->getDataPoint(n);
                newCenterCount[cluster]++;
            }
            
            // Compute actual means.
            for (int k = 0; k < K; k++)
            {
                if (newCenterCount[k] > 0)
                {
                    newCenters->getDataPoint(k) /= newCenterCount[k];
                }
                else
                {
                    newCenters->getDataPoint(k) = currentCenters->getDataPoint(k);
                }
            }
            
            // Compute the distance of each center to the new version.
            std::vector<float> newCenterDistances(N, 0);
            for (int k = 0; k < K; k++)
            {
                DataPoint difference = newCenters->getDataPoint(k)
                        - currentCenters->getDataPoint(k);
                
                newCenterDistances[k] = difference.transpose()*difference;
            }
            
            // Update lower bounds l(x, c):
            for (int n = 0; n < N; n++) 
            {
                for (int k = 0; k < K; k++)
                {
                    lowerBound(n, k) = std::max(lowerBound(n, k) - newCenterDistances[k], 0.f);
                }
            }
            
            // Update upper bounds u(x):
            for (int n = 0; n < N; n++)
            {
                const int cluster = currentLabels[n];
                currentDistances[n] = currentDistances[n] + newCenterDistances[cluster];
            }
            
            // Reset r(x):
            for (int n = 0; n < N; n++)
            {
                computePointClusterDistance[n] = true;
            }
            
            // Replace centers by new ones.
            currentCenters = newCenters;
        }
        
        // Compute the error:
        float error = 0;
        for (int n = 0; n < N; n++)
        {
            for (int k = 0; k < K; k++)
            {
                const int cluster = currentLabels[n];
                DataPoint difference = storage->getDataPoint(n) 
                        - currentCenters->getDataPoint(cluster);
                
                error += difference.transpose()*difference;
            }
        }
        
        if (error < bestError) {
            bestError = error;
            bestLabels = currentLabels;
            bestCenters = currentCenters;
        }
    }
    
    // Remember to actually return our findings!
    centers = bestCenters;
    labels = bestLabels;
    
    return bestError;
}

void KMeans::initCentersPP(AbstractDataStorage::ptr storage, 
        DataStorage::ptr centers)
{
    const int N = storage->getSize();
    const int K = numClusters;
    
    // K-means++ initialization.
    for (int k = 0; k < K; k++)
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

                    DataPoint difference = storage->getDataPoint(n) 
                            - centers->getDataPoint(c);
                    
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

void KMeans::initCentersRandom(AbstractDataStorage::ptr storage, 
        DataStorage::ptr centers)
{
    const int N = storage->getSize();
    const int K = numClusters;
    
    for (int k = 0; k < K; k++)
    {
        // Centers are chosen uniformly at random.
        int n = std::rand()%N;
        
        DataPoint center(storage->getDataPoint(n)); // Copy the center!
        centers->addDataPoint(center);
    }
}


////////////////////////////////////////////////////////////////////////////////
/// ClassStatisticsTool
////////////////////////////////////////////////////////////////////////////////

void ClassStatisticsTool::measure(AbstractDataStorage::ptr storage, std::vector<float> & result) const
{
    // Count the points
    result.resize(storage->getClasscount() + 1, 0.0f);
    
    for (int n = 0; n < storage->getSize(); n++)
    {
        if (storage->getClassLabel(n) != LIBF_NO_LABEL)
        {
            result[storage->getClassLabel(n)] += 1.0f;
        }
        else
        {
            // This data point has no label
            result[storage->getClasscount()] += 1.0f;
        }
    }
    
    // Normalize the distribution
    for (size_t c = 0; c < result.size(); c++)
    {
        result[c] /= storage->getSize();
    }
}

void ClassStatisticsTool::print(const std::vector<float> & result) const
{
    for (size_t c = 0; c < result.size(); c++)
    {
        printf("Class %3d: %4f%%\n", static_cast<int>(c), result[c]*100);
    }
}

void ClassStatisticsTool::measureAndPrint(AbstractDataStorage::ptr storage) const
{
    std::vector<float> result;
    measure(storage, result);
    print(result);
}
