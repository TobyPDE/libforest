#include "libforest/learning.h"
#include "libforest/util.h"
#include <iomanip>

using namespace libf;

////////////////////////////////////////////////////////////////////////////////
/// TreeLearnerState
////////////////////////////////////////////////////////////////////////////////

void TreeLearnerState::print()  
{
    float percentage = processed/static_cast<float>(total);
    if (total == 0)
    {
        percentage = 0;
    }
    
    std::cout << "Nodes: " << std::setw(6) << numNodes << std::endl;
    std::cout << "Depth: " << std::setw(6) << depth << std::endl;
}

void TreeLearnerState::reset()
{
    AbstractLearnerState::reset();
    total = 0;
    processed = 0;
    depth = 0;
    numNodes = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// ForestLearnerState
////////////////////////////////////////////////////////////////////////////////

void ForestLearnerState::print() 
{
    float percentageFinished = processed/static_cast<float>(total);
    float percentageStarted = startedProcessing/static_cast<float>(total);
    
    std::cout << "Finished trees: " << std::setw(4) << processed << '/' << total << std::endl;
    std::cout << "Started trees:  " << std::setw(4) << startedProcessing << '/' << total << std::endl;
}

void ForestLearnerState::reset()
{
    AbstractLearnerState::reset();
    total = 0;
    startedProcessing = 0;
    processed = 0;
}