# cat-SNE
Class-aware t-SNE: cat-SNE

----------

The python code catsne.py implements cat-SNE, a class-aware version of t-SNE, as well as quality assessment criteria for both supervised and unsupervised dimensionality reduction.

Cat-SNE was presented at the ESANN 2019 conference. 

Please cite as: 
- de Bodt, C., Mulders, D., López-Sánchez, D., Verleysen, M., & Lee, J. A. (2019). Class-aware t-SNE: cat-SNE. In ESANN (pp. 409-414).
- BibTeX entry:

@inproceedings{cdb2019catsne,

 title={Class-aware {t-SNE}: {cat-SNE}},
 
 author={de Bodt, C. and Mulders, D. and L\\'opez-S\'anchez, D. and Verleysen, M. and Lee, J. A.},
 
 booktitle={ESANN},
 
 pages={409--414},
 
 year={2019}
 
}

The most important functions of this file are:
- catsne: enables applying cat-SNE to reduce the dimension of a data set. The documentation of the function describes its parameters. 
- eval_dr_quality: enables evaluating the quality of an embedding in an unsupervised way. It computes quality assessment criteria measuring the neighborhood preservation from the high-dimensional space to the low-dimensional one. The documentation of the function explains the meaning of the criteria and how to interpret them.
- knngain: enables evaluating the quality of an embedding in a supervised way. It computes criteria related to the accuracy of a KNN classifier in the low-dimensional space. The documentation of the function explains the meaning of the criteria and how to interpret them.
- viz_qa: a plot function to easily visualize the quality criteria. 

At the end of the file, a demo presents how the code and the above functions can be used. Running this code will run the demo. Importing this module will not run the demo. 

Notations:
- DR: dimensionality reduction
- HD: high-dimensional
- LD: low-dimensional
- HDS: HD space
- LDS: LD space

References:
- [1] de Bodt, C., Mulders, D., López-Sánchez, D., Verleysen, M., & Lee, J. A. (2019). Class-aware t-SNE: cat-SNE. In ESANN (pp. 409-414).
- [2] Lee, J. A., & Verleysen, M. (2009). Quality assessment of dimensionality reduction: Rank-based criteria. Neurocomputing, 72(7-9), 1431-1443.
- [3] Lee, J. A., & Verleysen, M. (2010). Scale-independent quality criteria for dimensionality reduction. Pattern Recognition Letters, 31(14), 2248-2257.
- [4] Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013). Type 1 and 2 mixtures of Kullback–Leibler divergences as cost functions in dimensionality reduction based on similarity preservation. Neurocomputing, 112, 92-108.
- [5] Lee, J. A., Peluffo-Ordóñez, D. H., & Verleysen, M. (2015). Multi-scale similarities in stochastic neighbour embedding: Reducing dimensionality while preserving both local and global structure. Neurocomputing, 169, 246-261.
- [6] Maaten, L. V. D., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9(Nov), 2579-2605.
- [7] Jacobs, R. A. (1988). Increased rates of convergence through learning rate adaptation. Neural networks, 1(4), 295-307.

author: Cyril de Bodt (ICTEAM - UCLouvain)

@email: cyril __dot__ debodt __at__ uclouvain.be

Last modification date: May 15th, 2019

Copyright (c) 2019 Universite catholique de Louvain (UCLouvain), ICTEAM. All rights reserved.

This code was created and tested with Python 3.7.3 (Anaconda distribution, Continuum Analytics, Inc.). It uses the following modules:
- numpy: version 1.16.3 tested
- numba: version 0.43.1 tested
- scipy: version 1.2.1 tested
- matplotlib: version 3.0.3 tested
- scikit-learn: version 0.20.3 tested

You can use, modify and redistribute this software freely, but not for commercial purposes. 

The use of the software is at your own risk; the authors are not responsible for any damage as a result from errors in the software.
