import sys
from setuptools import setup, find_packages

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    sys.stderr.write("Build failure: 'pybind11' is required for the compilation phase.\n")
    sys.exit(1)

# Define compiler flags to guarantee optimal execution efficiency
COMPILER_FLAGS = ['-O3', '-march=native', '-ffast-math', '-std=c++17']

ext_modules = [
    Pybind11Extension(
        "bwsr_inference.core._core_models",
        sources=[
            "src/bwsr_inference/core/bindings.cpp",
            "src/bwsr_inference/core/models.cpp"
        ],
        extra_compile_args=COMPILER_FLAGS,
        language="c++"
    ),
]

setup(
    name="bwsr_inference",
    version="1.0.0",
    author="Dang-Khoa N. Nguyen",
    description="Bayesian Weak Signal Recovery Inference Pipeline",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    zip_safe=False,
)