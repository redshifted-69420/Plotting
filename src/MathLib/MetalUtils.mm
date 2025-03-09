#include "MetalUtils.h"

static id<MTLDevice> metalDevice = nil;
static id<MTLCommandQueue> metalCommandQueue = nil;
id<MTLLibrary> metalLibrary = nil;

void initializeMetal() {
  if (metalDevice == nil) {
    metalDevice = MTLCreateSystemDefaultDevice();
    if (!metalDevice) {
      std::cerr << "Error: Metal is not supported on this device!" << std::endl;
      return;
    }
    metalCommandQueue = [metalDevice newCommandQueue];

    NSError *error = nil;
    NSString *libraryPath = [NSString stringWithUTF8String:"default.metallib"];
    NSURL *libraryURL = [NSURL fileURLWithPath:libraryPath];
    metalLibrary = [metalDevice newLibraryWithURL:libraryURL error:&error];
    if (!metalLibrary) {
      std::cerr << "Error: Failed to load Metal library: "
                << error.localizedDescription.UTF8String << std::endl;
      return;
    }
  }
}