# GSEGUtils

GSEGUtils provides some tools and functionality that could be used across other development or research projects.
These are particularly focussed in the direction of research within the group.

The main goal if this repo is to help software processing efforts and allow more focus on algorithm 
development and research.

Currently this module includes:

| Package             | Description                                                      |
|---------------------|------------------------------------------------------------------|
| **ValidatedArray**  | Base classes with automated validation on type, shape and attributes |
| **Lazy Disk Cache** | Automated data offloading for memory management                  |
| **Util**            | Includes utility functions such as angle unit conversions        |
| **Validators**      | Helper functions for performing validation and normalisation     | 

## Installation

This can be done directly from the repository directory:

```commandline
pip install "GSEGUtils @ git+https://github.com/gseg-ethz/GSEGUtils.git"
```

or by cloning it

```commandline
git clone https://github.com/gseg-ethz/GSEGUtils.git
cd GSEGUtils
pip intstall .
 ```

## Usage
Accessing the different classes can then be easily done by importing from `GSEGUtils`:

```python
from GSEGUtils.base_array import BaseArry, Array_Nx3, Vector
import numpy

data = np.random.rand(10,20,3)
array = BaseArry(data.copy())
array += 10
assert np.all(array == data + 10)

coords3D = Array_Nx3(np.random.rand(20,3))
coords3d.arr = np.random.rand(100,3)    # Valid
coords3d.arr = data                     # RAISES ERROR -> invalid shape not of (N, 3)
```