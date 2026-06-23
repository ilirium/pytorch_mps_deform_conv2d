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
                    "-std=c++17",
                    "-ObjC++",
                    "-fobjc-arc",
                    # Embed the shader directory so the .mm can locate .metal at runtime.
                    "-DSHADER_DIR=\"deform_conv2d_mps/_C\"",
                ]
            },
            extra_link_args=[
                "-framework", "Metal",
                "-framework", "Foundation",
                "-framework", "MetalPerformanceShaders",
            ],
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
