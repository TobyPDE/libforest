#include "libforest/estimator_tools.h"

using namespace libf;

////////////////////////////////////////////////////////////////////////////////
/// GaussianKullbackLeiblerTool
////////////////////////////////////////////////////////////////////////////////

float GaussianKullbackLeiblerTool::measure(std::shared_ptr<EstimatorInterface> estimator, std::vector<Gaussian> & gaussians,
        const std::vector<float> & weights, int N)
{
    assert(weights.size() == gaussians.size());
    assert(gaussians.size() > 0);
    
    const int M = weights.size();
    
    float kl = 0;
    for (int n = 0; n < N; n++)
    {
        int m = std::rand()%M;
        DataPoint x;
        gaussians[m].sample(x);

        float p_x = 0;
        float p_x_hat = estimator->estimate(x);
        
        for (m = 0; m < M; m++)
        {
            p_x += weights[m]*gaussians[m].evaluate(x);
        }
        
        if (p_x > 0)
        {
            kl += fastlog2(p_x_hat/p_x);
        }
    }
    
    return kl/N;
}

void GaussianKullbackLeiblerTool::print(float kl)
{
    printf("Divergence: %2.2f\n", kl);
}

void GaussianKullbackLeiblerTool::measureAndPrint(std::shared_ptr<EstimatorInterface> estimator, std::vector<Gaussian> & gaussians,
        const std::vector<float> & weights, int N)
{
    float kl = measure(estimator, gaussians, weights, N);
    print(kl);
}

////////////////////////////////////////////////////////////////////////////////
/// GaussianSquaredErrorTool
////////////////////////////////////////////////////////////////////////////////

float GaussianSquaredErrorTool::measure(std::shared_ptr<EstimatorInterface> estimator, std::vector<Gaussian> & gaussians,
        const std::vector<float> & weights, int N)
{
    assert(weights.size() == gaussians.size());
    assert(gaussians.size() > 0);
    
    const int M = weights.size();
    
    float se = 0;
    for (int n = 0; n < N; n++)
    {
        int m = std::rand()%M;
        DataPoint x;
        gaussians[m].sample(x);

        float p_x = 0;
        float p_x_hat = estimator->estimate(x);
        
        for (m = 0; m < M; m++)
        {
            p_x += weights[m]*gaussians[m].evaluate(x);
        }
        
        se += (p_x - p_x_hat)*(p_x - p_x_hat);
    }
    
    return se/N;
}

void GaussianSquaredErrorTool::print(float se)
{
    printf("Error: %2.6f\n", se);
}

void GaussianSquaredErrorTool::measureAndPrint(std::shared_ptr<EstimatorInterface> estimator, std::vector<Gaussian> & gaussians,
        const std::vector<float> & weights, int N)
{
    float se = measure(estimator, gaussians, weights, N);
    print(se);
}