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
    return min + static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * (max - min);
}

cv::Mat visualizeGaussians(int H, int W, std::vector<Gaussian> gaussians, std::vector<float> weights)
{
    assert(weights.size() == gaussians.size());
    const int M = weights.size();
    
    cv::Mat image(H, W, CV_32FC1, cv::Scalar(0));
    float p_max = 0;
    
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            DataPoint x(2);
            x.at(0) = i;
            x.at(1) = j;
            
            float p_x = 0;
            
            for (int m = 0; m < M; m++)
            {
                p_x += weights[m]*gaussians[m].evaluate(&x);
            }
            
            if (p_x > p_max)
            {
                p_max = p_x;
            }
            
            image.at<float>(i, j) = p_x;
        }
    }
    
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            image.at<float>(i, j) = image.at<float>(i, j)/p_max * 255;
        }
    }
    
    return image;
}

cv::Mat visualizeSamples(int H, int W, const UnlabeledDataStorage & storage)
{
    cv::Mat image(H, W, CV_8UC1, cv::Scalar(255));
    
    for (int n = 0; n < storage.getSize(); n++)
    {
        DataPoint* x = storage.getDataPoint(n);
        
        int i = std::floor(x->at(0));
        int j = std::floor(x->at(1));
        
        if (i >= 0 && i < H && j >= 0 && j < W)
        {
            image.at<unsigned char>(i, j) = 0;
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
    
    // Number of components.
    const int M = 3;
    const int H = 400;
    const int W = 400;
    
    // New seed.
    std::srand(std::time(0));
    
    std::vector<Gaussian> gaussians;
    std::vector<float> weights(M);
    float weights_sum = 0;
    
    for (int m = 0; m < M; m++)
    {
        do
        {
            weights[m] = randFloat(0, 1);
        }
        while (weights[m] == 0);
        
        weights_sum +=weights[m];
        
        float v0 = randFloat(25, H/4);
        float v1 = randFloat(25, W/4);
        float theta = randFloat(0, M_PI);
        
        Eigen::Matrix2f covariance = genCovar(v0, 2*v1, theta);
        
        Eigen::Vector2f mean(2);
        mean(0) = randFloat(H/4, 3*(H/4));
        mean(1) = randFloat(W/4, 3*(W/4));
        
        Gaussian gaussian;
        gaussian.setMean(mean);
        gaussian.setCovariance(covariance);
        
        gaussians.push_back(gaussian);
    }
    
    for (int m = 0; m < M; m++)
    {
        weights[m] /= weights_sum;
    }
    
    cv::Mat image = visualizeGaussians(H, W, gaussians, weights);
    cv::imwrite("gaussians.png", image);
    
    // Generate samples.
    const int N = parameters["num-samples"].as<int>();
    UnlabeledDataStorage storage;
    
    for (int n = 0; n < N; n++)
    {
        int m = std::rand() % M;
        storage.addDataPoint(gaussians[m].sample());
    }
    
    cv::Mat image_samples = visualizeSamples(H, W, storage);
    cv::imwrite("samples.png", image_samples);    
    
    DensityTreeLearner learner;
    learner.addCallback(DensityTreeLearner::defaultCallback, 1);
    
    DensityTree* tree = learner.learn(&storage);
    
    GMMDensityAccuracyTool accuracyTool;
    accuracyTool.measureAndPrint(tree, gaussians, weights, N);
    
    UnlabeledDataStorage storage_tree;
    for (int n = 0; n < N; n++)
    {
        storage_tree.addDataPoint(tree->sample());
    }
    
    cv::Mat image_tree = visualizeSamples(H, W, storage_tree);
    
    delete tree;
    return 0;
}
