#ifndef METAL_UTILS_H
#define METAL_UTILS_H

// Include necessary headers
#import <Metal/Metal.h> // Include Metal headers for Objective-C
#include <iostream> // For std::cerr

// Declare the metalLibrary variable
extern id<MTLLibrary> metalLibrary;

// Declare the initializeMetal function
void initializeMetal();

#endif // METAL_UTILS_H
