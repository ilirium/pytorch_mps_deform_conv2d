"""Diagnostic: is the MPS kernel for add_one actually registered?

Run:  python tests/diag_dispatch.py
Paste the full output back.
"""

import os

# OMP Error #15 workaround (duplicate OpenMP runtimes in conda envs);
# must be set before `import torch`. See Makefile.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

# Import the compiled extension directly so its static initializers
# (TORCH_LIBRARY / TORCH_LIBRARY_IMPL) run.
import deform_conv2d_mps._C_ext as ext  # noqa: F401

print("torch:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())
print("_C_ext file:", getattr(ext, "__file__", "<built-in>"))

has_ns = hasattr(torch.ops, "deform_conv2d_mps")
print("namespace deform_conv2d_mps present:", has_ns)
print("add_one attr present:", hasattr(torch.ops.deform_conv2d_mps, "add_one"))

print("\n--- dispatch registrations for deform_conv2d_mps::add_one ---")
try:
    print(torch._C._dispatch_dump("deform_conv2d_mps::add_one"))
except Exception as e:  # noqa: BLE001
    print("dispatch_dump failed:", repr(e))

print("\n--- computed dispatch table ---")
try:
    print(torch._C._dispatch_dump_table("deform_conv2d_mps::add_one"))
except Exception as e:  # noqa: BLE001
    print("dispatch_dump_table failed:", repr(e))
