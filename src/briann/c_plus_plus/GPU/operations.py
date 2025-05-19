from __future__ import annotations

from pybind11_builts.GPU_operations_module import __doc__, __version__, add, subtract

# Setting this special variables will import the below symbols whenever the user writes from <module> import *
__all__ = ["__doc__", "__version__", "add", "subtract"]