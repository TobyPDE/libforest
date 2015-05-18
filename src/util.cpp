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

void GUIUtil::printProgressBar(float percentage, int size)
{
    for (int k = 0; k < size; k++)
    {
        const float p = (k+1)/static_cast<float>(size);
        if (p <= percentage)
        {
            std::cout << '#';
        }
        else
        {
            std::cout << ' ';
        }
    }
}