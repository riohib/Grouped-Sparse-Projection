# GSP

### Python source files for the paper: Grouped Sparse Projection.

Authors: Nicolas Gillis, Riyasat Ohib, Sergey Plis, Vamsi Potluru.

In this paper, we design a new sparse projection method for a set of vectors in order to achieve a desired average level of sparsity which is measured using the ratio of the l1 and l2 norms.

This work supersedes the work by 

Vamsi K. Potluru, Sergey M. Plis, Jonathan Le Roux, Barak A. Pearlmutter, Vince D. Calhoun, Thomas P. Haye.

available here: https://github.com/ismav/sparseNMF

```
./                                - Top directory.
./README.md                       - This is the readme file.
./license.md                      - The license for the repository.
|LeNet_300_100/                   - Contains the python source files for the Sparse LeNet 300-100 experiments.
    |--- LeNet300.py              - main driver function. Run these to generate results.
    |--- gs_projection.py         - Contains the GSP algorithm.
    |--- helper.py                - contains some helper functions for various calculations.
    |--- readme.md                - specific readme for the directory.
    
|Sparse_Autoencoders/   - Contains the python source files for the Sparse Autoencoder experiments.
    |--- main.py.                 - main driver function. Run these to generate results.
    |--- gs_projection.py         - Contains the GSP algorithm.
    |--- AutoencoderSimple.py     - Contains the Autoencoder model Class.
    |--- helper.py                - contains some helper functions for various calculations.
    |--- readme.md                - specific readme for the directory.
    
