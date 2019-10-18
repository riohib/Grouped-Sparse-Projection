# GSP
Group Sparse Projection

main.py contains the training loop and data processing.

Autoencoder.py :
Contains the autoencoder class and also calls the sparsity from the projections.py
The sparsity (projection) is called from the "forward" method within the class.

I also tried projection in the training loop. 
That can be tried by uncommenting the sparsity block in the training loop,
and commenting the sparsity block in forward method.
