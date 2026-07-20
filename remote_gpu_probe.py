import importlib

for name in ["torch", "numpy", "pandas", "pyarrow", "sklearn", "scipy", "matplotlib"]:
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "ok"))
    except Exception as exc:
        print(name, "MISSING", repr(exc))

import torch
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("cuda_runtime", torch.version.cuda)
