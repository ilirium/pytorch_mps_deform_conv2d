"""Build script for the native MPS extension.

The extension is a single Objective-C++ (.mm) translation unit that registers
the `deform_conv2d_mps` ops with PyTorch and dispatches Metal compute shaders.

Only builds the native extension on macOS. On other platforms the package still
installs (Python-only) and falls back to the torchvision CPU reference.
"""

import platform

from setuptools import setup

ext_modules = []
cmdclass = {}

if platform.system() == "Darwin":
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    ext_modules = [
        CppExtension(
            name="deform_conv2d_mps._C_ext",
            sources=["src/deform_conv2d_mps/_C/deform_conv2d_mps.mm"],
            extra_compile_args={
                "cxx": [
                    # torch >= 2.14 headers use C++20 features (bit-field
                    # default member initializers); c++17 builds fine but
                    # spews -Wc++20-extensions warnings from torch includes.
                    "-std=c++20",
                    "-ObjC++",
                    "-fobjc-arc",
                    # MPS APIs require macOS 13+; also silences availability warnings
                    # when the conda toolchain defaults the target to 11.0.
                    "-mmacosx-version-min=13.0",
                ]
            },
            extra_link_args=[
                "-mmacosx-version-min=13.0",
                "-framework", "Metal",
                "-framework", "Foundation",
            ],
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
