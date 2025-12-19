# POMDP Framework and Compilation Infrastructure

A general framework for designing and implementing Partially Observable Markov Decision Process (POMDP) models with a flexible compilation infrastructure that targets various computing devices.

## Overview

This project provides a comprehensive POMDP framework that enables researchers and developers to design POMDP models and compile them efficiently for deployment across different hardware platforms. The framework abstracts the complexity of device-specific optimizations while maintaining the mathematical rigor required for POMDP planning and execution.

## Features

- **General POMDP Model Design**: Flexible API for defining states, actions, observations, and transition/observation functions
- **Multi-Device Compilation**: Compile POMDP models to various target devices including CPUs, GPUs, and embedded systems
- **Modular Architecture**: Clean separation between model definition and compilation infrastructure
- **Extensible Backend**: Easy integration of new compilation targets and optimization passes

## Building the Project

The project uses CMake as its build system. Follow these steps to build:

### Prerequisites

- CMake 3.15 or higher
- A C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)

### Build Instructions

```bash
# Clone the repository
git clone <repository-url>
cd pomdp-framework

# Create a build directory
mkdir build
cd build

# Configure the project
cmake ..

# Build examples. replace the <example name> to a specific example
make <example name> 
```

### Build Options

You can customize the build with the following CMake options:

```bash
# Example: Build with specific compiler
cmake -DCMAKE_CXX_COMPILER=g++ ..

# Example: Build in Release mode
cmake -DCMAKE_BUILD_TYPE=Release ..
```

## Project Status

This project is under active development. Current focus areas include:

- Core POMDP model representation
- Simple POMCP for continuous belief state and action space.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is currently under MIT license.
