from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import sys

# Function to define PyTorch version-dependent macros
def get_version_dependent_macros():
    """Define macros based on PyTorch version for compatibility."""
    macros = []
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
        macros.append('-DVERSION_GE_1_1')
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
        macros.append('-DVERSION_GE_1_3')
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
        macros.append('-DVERSION_GE_1_5')
    return macros

# Get version-dependent macros
version_macros = get_version_dependent_macros()

# Define compiler flags for C++ and NVCC
extra_compile_args = {
    'cxx': ['-O3', '-std=c++17'] + version_macros,  # C++ flags
    'nvcc': [
        '-O3',              # Optimization level
        '-lineinfo',        # For debugging CUDA code
        '--use_fast_math',  # Optimize floating-point operations
        '--std=c++17',      # C++17 standard for NVCC
        '-U__CUDA_NO_HALF_OPERATORS__',       # Enable half-precision operators
        '-U__CUDA_NO_HALF_CONVERSIONS__',     # Enable half-precision conversions
        '-U__CUDA_NO_HALF2_OPERATORS__'       # Enable half2 operators
    ] + version_macros  # Add version macros to NVCC as well
}

# Setup configuration
setup(
    name="fused_adam",
    version="0.0.1",
    author="JosephLiaw",
    description="Fused Adam optimizer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name="fused_adam.fused_adam_frontend",
            sources=[
                'src/fused_adam/fused_adam_frontend.cpp',
                'src/fused_adam/multi_tensor_adam.cu'
            ],
            include_dirs=["src/fused_adam/includes"],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"]
)