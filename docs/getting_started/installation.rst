Installation
============

Requirements
------------
- Python 3.10 or newer
- A recent Rust toolchain when building from source

Install from PyPI
-----------------
.. code-block:: bash

   pip install medrs

Optional extras:
.. code-block:: bash

   pip install "medrs[torch]"      # PyTorch helpers
   pip install "medrs[monai]"      # MONAI integration
   pip install "medrs[jax]"        # JAX helpers

Development Install
-------------------
.. code-block:: bash

   git clone https://github.com/medrs/medrs.git
   cd medrs
   pip install -e ".[dev]"
   maturin develop --features python

Verification
------------
.. code-block:: python

   import medrs
   print("medrs version:", medrs.__version__)

CUDA Support
------------
Install a CUDA-enabled PyTorch or JAX build that matches your GPU and driver. For PyTorch:
.. code-block:: bash

   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

If CUDA is unavailable, install the appropriate wheel from https://pytorch.org.
