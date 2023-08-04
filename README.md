# Refractored_star_classifier 

Refactored version of the classifier presented in [arxiv:1605.03201][original_classifier]. Used in conjunction with [WaveDiff][wavediff_repo] with the intention of improving PSF quality by providing a higher number of stars with approximate SEDs.

dataset_generation provides the code necessary to generate coherent datasets in a CPU cluster and then visualize the generated outputs.

configurations contains various notebooks to train and test different classifier configurations. single_classifier contains the original archictecture and committee classifier implements that architecture in a classifyng committee of 48 networks. The most variable notebook with different number of PCA entries and different architectures is hyperparameter_dependance. softmax classifier uses softmax activation in the last layer instead of the single output of the regression parameter. CNN_classifier takes this one step further by replacing the inputs of the network by the full image instead of its PCA components and also changing the architecture into a convolutional network of course.

dataset_manipulation allows to modify the datasets in different ways. PCA generates the PCA inputs needed for the classifier. pre_WD_classifier and pre_WD_classifier_CNN classify the stars in a dataset and assign them the according SEDs and stellar classes, the former with an architechture that uses PCA inputs and the latter with a CNN architecture. pre_WD_true+CNN takes a dataset and classifies the rest of stars in the field of view not included in the dataset.

[original_classifier]: https://arxiv.org/abs/1605.03201
[wavediff_repo]: https://github.com/CosmoStat/wf-psf