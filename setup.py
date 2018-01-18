# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 09:45:36 2018

@author: nfitch3
"""

from setuptools import setup, find_packages
from setuptools.command.install import install

package = 'newsgraphing'
version = '0.1'
license='MIT',
long_description=open('README.md').read(),

# from setuptools.command.develop import develop


# class PostDevelopCommand(develop):
#     """Post-installation for development mode."""
#     def run(self):
#         # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
#         develop.run(self)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        import nltk
        nltk.download('punkt')
        install.run(self)

print(find_packages('src'))
setup(name=package,
      version=version,
      packages=['newsgraphing'],
      install_requires=[
          'numpy',
          'networkx',
          'matplotlib',
          'scipy',
          'pandas',
          'nltk',
          'feather-format',
          'gensim',
          'imblearn',
          'ipywidgets',
          'psycopg2',
          'pymongo',
          'scikit_learn',
          'spacy',
          'tldextract',
          'tqdm',
          'vaderSentiment'],
      package_dir={'': 'src'},
      description="Analyzes news articles for bias and credibility",
      url='url',
      cmdclass={
	      # 'develop': PostDevelopCommand,
	      'install': PostInstallCommand,
      }
)

