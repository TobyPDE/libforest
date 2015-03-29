#include "libforest/util.h"
#include <random>
#include <iomanip>

static std::random_device rd;

using namespace libf;

////////////////////////////////////////////////////////////////////////////////
/// Util
////////////////////////////////////////////////////////////////////////////////

void Util::generateRandomPermutation(int N, std::vector<int> & sigma)
{
    // Set up the initial state
    sigma.resize(N);
    for (int n = 0; n < N; n++)
    {
        sigma[n] = n;
    }
    
    // Randomize the permutation
    std::shuffle(sigma.begin(), sigma.end(), std::default_random_engine(rd()));
}

////////////////////////////////////////////////////////////////////////////////
/// GUIUtil
////////////////////////////////////////////////////////////////////////////////

void GUIUtil::printProgressBar(float percentage)
{
    // We set the width of the bar to 60 characters
    int mcol = 60;
    
    // Calculate the number of progress bar segments
    // 8 = 2 spacers, 4 characters for the percentage, 1 blank space, 1 line feed
    int progressBarWidth = mcol - 8;
    
    std::cout << '[';
    
    for (int k = 0; k < progressBarWidth; k++)
    {
        const float p = k/static_cast<float>(progressBarWidth);
        if (p <= percentage)
        {
            std::cout << '=';
        }
        else
        {
            std::cout << ' ';
        }
    }
    
    std::cout << "] " << std::setw(4) << percentage * 100 << '%' << std::endl;
}