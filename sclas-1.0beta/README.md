## Summary

A beta version of the _sclas_ package (stellar classifier). A simple algorithm that builds on the
fact that, in diffraction-limited images, the resolution depends on the wavelength.
Thus, objects whose size are similar to the Point Spread Function (PSF) or small will differ in
shape if they do not share the same spectra.

## Peer-reviewed article

For much more information, and a application to classification of stellar objects by their spectral
classes, please refer to 

[T. Kuntzer, M. Tewes and F. Courbin, 2016, A&A, 591, A54](doi:10.1051/0004-6361/201628660)

## Requirements

The following packages are needed:

- scipy
- numpy
- matplotlib
- scikit-learn

Optional requirements:

- [FANN](http://leenissen.dk/fann/wp/) (including its Python bindings)
- [SkyNet](https://ccpforge.cse.rl.ac.uk/gf/project/skynet/) (including its Python bindings)
- Inspect

Please note that sclas is meant to be run in a python2.x environment.

## Installation

This is a beta version, however, you can install the module using the standard:

    sudo python setup.py install

to remove type

    sudo pip uninstall sclas

To install it temporarily in a terminal, the absolute path to the sclas direction must be 
added to the python path,

    export PYTHONPATH=${PYTHONPATH}:/absolute/path/to/sclas/

This must point to the folder containing the sclas directory.

To test the successful exportation type

    import sclas

in any python interpreter.

## Demonstrations

To get a feeling of the usage of the code, you can visit the demo/ directory which contains
different configurations of sclas to be run on the [MNIST](http://yann.lecun.com/exdb/mnist/) 
dataset (which are included in this scikit-learn package) and run 

    python digits.py

or/and

    python digits_committees.py


