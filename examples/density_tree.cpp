#include <iostream>
#include "libforest/libforest.h"
#include <chrono>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace libf;

Eigen::Matrix2f genCovar(float v0, float v1, float theta)
{
    Eigen::Matrix2f rot = Eigen::Rotation2Df(theta).matrix();
    return rot * Eigen::DiagonalMatrix<float, 2, 2>(v0, v1) * rot.transpose();
}

float randFloat(float min, float max)
{
    return min + static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX/max));
}

cv::Mat visualizeGaussians(int H, int W, std::vector<Gaussian> gaussians, std::vector<float> weights)
{
    assert(weights.size() == gaussians.size());
    
    cv::Mat image(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            float p_x = 0;
            
            DataPoint x(2);
            x.at(0) = i;
            x.at(1) = j;
            
            for (int m = 0; m < weights.size(); m++)
            {
                p_x += weights[m] * gaussians[m].evaluate(&x);
            }
            
            image.at<cv::Vec3b>(i, j) = cv::Vec3b(0, p_x*255, 255);
        }
    }
    
    return image;
}

/**
 * Example of density tree.
 * 
 * Usage:
 * $ ./lib_forest/examples/cli_decision_tree --help
 * Allowed options:
 *   --help                   produce help message
 *   --num-features arg (=10) number of features to use (set to dimensionality of 
 *                            data to learn deterministically)
 *   --max-depth arg (=100)   maximum depth of trees
 */
int main(int argc, const char** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("num-samples", boost::program_options::value<int>()->default_value(10000),"number of samples for training")
        ("num-features", boost::program_options::value<int>()->default_value(10), "number of features to use (set to dimensionality of data to learn deterministically)")
        ("max-depth", boost::program_options::value<int>()->default_value(100), "maximum depth of trees");

    boost::program_options::positional_options_description positionals;
    
    boost::program_options::variables_map parameters;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionals).run(), parameters);
    boost::program_options::notify(parameters);

    if (parameters.find("help") != parameters.end())
    {
        std::cout << desc << std::endl;
        return 1;
    }
    
    // Generate a artificial dataset from a gaussian mixture model.
    const int N = parameters["num-samples"].as<int>();
    const int M = 2;
    const int W = 400;
    const int H = 400;
    
    std::vector<float> w(M, 0.f);
    float w_sum = 0.f;
    
    std::vector<Gaussian> gaussians(M);
    
    for (int m = 0; m < M; m++)
    {
        w[m] = std::rand()%100;
        w_sum += w[m];
        
        Eigen::VectorXf mean;
        mean(0) = randFloat(H/4.f, H/4.f + H/2.f);
        mean(1) = randFloat(W/4.f, W/4.f + W/2.f);
        
        float v0 = randFloat(0.f, H/100.f);
        float v1 = randFloat(0.f, W/100.f);
        float theta = std::fmod(randFloat(0.f, 10.f), M_PI);
        
        Eigen::MatrixXf covariance = genCovar(v0, v1, theta);
        
        Gaussian gaussian(mean, covariance);
        gaussians[m] = gaussian;
    }
    
    // Normalize weights.
    std::vector<float> w_cum_sum(M, w[0]);
    for (int m = 0; m < M; m++)
    {
        w[m] /= w_sum;
        
        // Update cumulative sum.
        if (m > 0)
        {
            w_cum_sum[m] += w_cum_sum[m - 1];
        }
    }
    
    UnlabeledDataStorage storage;
    for (int n = 0; n < N; n++)
    {
        float r = randFloat(0.f, 1.f);
        int m = 0;
        while (r > w_cum_sum[m])
        {
            m++;
        }
        
        assert(m < M);
        
        // Sample from the m-th gaussian.
        DataPoint* x = gaussians[m].sample();
        storage.addDataPoint(x);
    }
    
    // Visualize the initial distribution as image.
    cv::Mat image = visualizeGaussians(H, W, gaussians, w);
    cv::imwrite("test.png", image);
    return 0;
}
