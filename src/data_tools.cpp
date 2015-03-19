#include "libforest/data_tools.h"
#include "libforest/io.h"

using namespace libf;

////////////////////////////////////////////////////////////////////////////////
/// KMeans
////////////////////////////////////////////////////////////////////////////////

float computeDistance(const DataPoint & x, const DataPoint & y)
{
    const int D = x.rows();
    
    float difference = 0;
    float distance = 0;
    
    // Checking whether this is more efficient than using Eigen.
    for (int d = 0; d < D; d++)
    {
        // Saves a single difference!
        difference = x(d) - y(d);
        
        distance += difference*difference;
    }
    
    return distance;
}

float KMeans::cluster(AbstractDataStorage::ptr storage, 
        AbstractDataStorage::ptr centers, std::vector<int> & labels)
{
    BOOST_ASSERT_MSG(storage->getSize() > 0, "Cannot run k-means on an empty data storage.");
    
    const int N = storage->getSize();
    const int D = storage->getDimensionality();
    const int K = numClusters;
    const int T = numIterations;
    const int M = numTries;

    // To identify the best clustering.
    float error = 1e35;
    
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
        std::vector<float> upperBound(N, 1e35);
        
        // l(x, c) in [1]:
        Eigen::MatrixXf lowerBound(N, K);
        
        // Distance amtrix between centers.
        Eigen::MatrixXf clusterDistances(K, K);
        
        // s(c) in [1):
        std::vector<float> minClusterDistance(K, 1e35);
        
        for (int k = 0; k < K; k++)
        {
            for (int l = 0; l < K; l++)
            {
                clusterDistances(k, l) = computeDistance(currentCenters->getDataPoint(k), 
                        currentCenters->getDataPoint(l));
            }
            
            // Assign point to nearest cluster:
            for (int n = 0; n < N; n++)
            {
                // Set the lower bound l(x, c):
                // Previously: assign to local variable rather than l(x, c).
                lowerBound(n, k) = computeDistance(storage->getDataPoint(n), 
                        currentCenters->getDataPoint(k));

                if (lowerBound(n, k) < upperBound[n])
                {
                    // Update upper bound u(x):
                    upperBound[n] = lowerBound(n, k);

                    // Assign initial label.
                    currentLabels[n] = k;
                }
            }
        }
        
        // r(x) in [1]:
        std::vector<bool> computePointClusterDistance(N, true);
        
        // Used to compute new centers each iteration:
        DataStorage::ptr newCenters = DataStorage::Factory::create();
        std::vector<int> newCenterCount(K, 0);

        // Initialize data storage.
        for (int k = 0; k < K; k++)
        {
            newCenters->addDataPoint(Eigen::VectorXf::Zero(D));
        }
        
        // Holds the distances d(c, m(c)) between center and new center.
        std::vector<float> newCenterDistances(N, 0);
        
        for (int t = 0; t < T; t++)
        {
            // Update the between center distances:
            for (int k = 0; k < K; k++)
            {
                for (int l = 0; l < K; l++)
                {
                    clusterDistances(k, l) = computeDistance(currentCenters->getDataPoint(k), 
                            currentCenters->getDataPoint(l));
                    
                    // Compute distance to nearest cluster:
                    if (clusterDistances(k, l) < minClusterDistance[k])
                    {
                        minClusterDistance[k] = clusterDistances(k, l);
                    }
                }
            }
            
            for (int k = 0; k < K; k++)
            {
                for (int n = 0; n < N; n++)
                {
                    // Current cluster of this point.
                    const int cluster = currentLabels[n];
                    
                    // If u(x) <= 1/2 * s(c(x)), then skip this point.
                    if (upperBound[n] <= 1.f/2.f * minClusterDistance[cluster])
                    {
                        continue;
                    }
                    
                    if (k == cluster) 
                    {
                        continue;
                    }
                    
                    if (upperBound[n] <= lowerBound(n, k))
                    {
                        continue;
                    }
                    
                    if (upperBound[n] <= 1.f/2.f * clusterDistances(cluster, k))
                    {
                        continue;
                    }
                    
                    float distance  = 0;
                    if (computePointClusterDistance[n])
                    {
                        // We need to compute the difference to the current cluster.
                        distance = computeDistance(storage->getDataPoint(n), 
                                currentCenters->getDataPoint(cluster));
                        
                        // Update lower bound l(x, c):
                        lowerBound(n, k) = distance;
                        
                        // Update upper bound u(x):
                        upperBound[n] = distance;
                        
                        computePointClusterDistance[n] = false;
                    }
                    else
                    {
                        distance = upperBound[n];
                    }
                    
                    if (distance > lowerBound(n, k) 
                            || distance > 1.f/2.f * clusterDistances(cluster, k))
                    {
                        // Compute difference to new center.
                        // Previously: save to local variable instead of l(x, c).
                        // Update l(x, c) at the same time.
                        lowerBound(n, k) = computeDistance(storage->getDataPoint(n), 
                                currentCenters->getDataPoint(k));
                        
                        if (lowerBound(n, k) < distance)
                        {
                            // Update upper bound u(x):
                            upperBound[n] = lowerBound(n, k);
                            
                            currentLabels[n] = k;
                        }
                    }
                }
            }
            
            std::fill(newCenterCount.begin(), newCenterCount.end(), 0);
            
            // Compute center means.
            for (int n = 0; n < N; n++)
            {
                const int cluster = currentLabels[n];
                
                if (n == 0)
                {
                    newCenters->getDataPoint(cluster) = storage->getDataPoint(n);
                }
                else
                {
                    newCenters->getDataPoint(cluster) += storage->getDataPoint(n);
                }
                
                newCenterCount[cluster]++;
            }
            
            // Compute actual means.
            for (int k = 0; k < K; k++)
            {
                if (newCenterCount[k] > 0)
                {
                    newCenters->getDataPoint(k) /= newCenterCount[k];
                    
                    // Compute distance to new center.
                    newCenterDistances[k] = computeDistance(newCenters->getDataPoint(k),
                            currentCenters->getDataPoint(k));
                }
                else
                {
                    newCenters->getDataPoint(k) = currentCenters->getDataPoint(k);
                    
                    // Nothing changed, so distance is zero!
                    newCenterDistances[k] = 0;
                }
            }
            
            for (int n = 0; n < N; n++) 
            {
                // Update lower bounds l(x, c):
                for (int k = 0; k < K; k++)
                {
                    lowerBound(n, k) = std::max(lowerBound(n, k) - newCenterDistances[k], 0.f);
                }
                
                const int cluster = currentLabels[n];
                
                // Update upper bound u(x):
                upperBound[n] = upperBound[n] + newCenterDistances[cluster];
                
                // Update r(x):
                computePointClusterDistance[n] = true;
            }
            
            // Replace centers by new ones.
            currentCenters = newCenters;
        }
        
        // Compute the error:
        float currentError = 0;
        for (int n = 0; n < N; n++)
        {
            const int cluster = currentLabels[n];
            
            currentError += computeDistance(storage->getDataPoint(n), 
                    currentCenters->getDataPoint(cluster));
        }
        
        if (currentError < error) {
            error = currentError;
            labels = currentLabels;
            centers = currentCenters;
        }
    }
    
    return error;
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

////////////////////////////////////////////////////////////////////////////////
/// ZScoreNormalizer
////////////////////////////////////////////////////////////////////////////////

void ZScoreNormalizer::learn(AbstractDataStorage::ptr storage)
{
    const int D = storage->getDimensionality();
    const int N = storage->getSize();
    
    // Reset the model
    mean = DataPoint::Zero(D);
    stdev = DataPoint::Zero(D);
    
    // Compute the mean of the data set
    for (int n = 0; n < N; n++)
    {
        mean += storage->getDataPoint(n);
    }
    
    // Normalize the mean
    mean /= N;
    
    // Compute the standard deviation
    for (int n = 0; n < N; n++)
    {
        const DataPoint temp = storage->getDataPoint(n) - mean;
        stdev += temp.cwiseProduct(temp);
    }
    
    // Normalize
    stdev /= N;
    
    // Compute the square root
    for (int d = 0; d < D; d++)
    {
        stdev(d) = std::sqrt(stdev(d));
    }
}

void ZScoreNormalizer::apply(DataStorage::ptr storage) const
{
    const int D = storage->getDimensionality();
    const int N = storage->getSize();
    
    BOOST_ASSERT_MSG(D == mean.rows(), "Mismatch between the learned model and the given data storage.");
    
    for (int n = 0; n < N; n++)
    {
        storage->getDataPoint(n) = (storage->getDataPoint(n) - mean).cwiseProduct(stdev.cwiseInverse());
    }
}

void ZScoreNormalizer::read(std::istream& stream)
{
    readBinary(stream, mean);
    readBinary(stream, stdev);
}

void ZScoreNormalizer::write(std::ostream& stream) const
{
    writeBinary(stream, mean);
    writeBinary(stream, stdev);
}


////////////////////////////////////////////////////////////////////////////////
/// PCA
////////////////////////////////////////////////////////////////////////////////

void PCA::learn(AbstractDataStorage::ptr storage)
{
    const int D = storage->getDimensionality();
    const int N = storage->getSize();
    
    // Set up the data matrix
    Eigen::MatrixXf X = Eigen::MatrixXf::Zero(D, N);
    
    // Compute the mean of the data points
    mean = DataPoint::Zero(D);
    for (int n = 0; n < N; n++)
    {
        mean += storage->getDataPoint(n);
    }
    mean /= N;
    
    for (int n = 0; n < N; n++)
    {
        X.col(n) = storage->getDataPoint(n) - mean;
    }
    
    // Compute the singular value decomposition
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(X, Eigen::ComputeFullU);
    
    V = svd.matrixU();
}

void PCA::apply(DataStorage::ptr storage, int M) const
{
    BOOST_ASSERT_MSG(1 <= M && M <= V.rows(), "Invalid number of projection dimensions.");
    
    const int N = storage->getSize();
    
    for (int n = 0; n < N; n++)
    {
        const DataPoint temp = storage->getDataPoint(n) - mean;
        storage->getDataPoint(n).resize(M, 1);
        storage->getDataPoint(n) = V.topRows(M)*temp;
    }
}

void PCA::read(std::istream& stream)
{
    readBinary(stream, V);
    readBinary(stream, mean);
}

void PCA::write(std::ostream& stream) const
{
    writeBinary(stream, V);
    writeBinary(stream, mean);
}