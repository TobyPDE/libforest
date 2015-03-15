#include "util.h"

static std::random_device rd;

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
