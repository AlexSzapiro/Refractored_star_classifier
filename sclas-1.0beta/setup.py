from setuptools import setup

setup(name='sclas',
      version='1.0b',
      description='Aggregation of machine learning codes to do reduce the dimensionality of images and do their classification.',
      authors='Thibault Kuntzer, Malte Tewes, Frederic Courbin',
      author_email='thibault.kuntzer@epfl.ch',
      packages=['sclas','sclas.classifier', 'sclas.diagnostics', 'sclas.encoder', 'sclas.plots'],
      zip_safe=False)
