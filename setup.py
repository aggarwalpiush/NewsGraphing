# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 09:45:36 2018

@author: nfitch3
"""

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

package = 'newsgraphing'
version = '0.1'
license='MIT',
long_description=open('README.md').read(),

# from setuptools.command.develop import develop

def installNLP():
    try:
        import nltk
        from spacy.cli import download as sdl
        nltk.download('punkt')
        sdl('download', 'en')
    except Exception as ex:
        print("Cannot download spacy or nltk models.")
        raise ex

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        installNLP()
        develop.run(self)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        installNLP()
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
	      'develop': PostDevelopCommand,
	      'install': PostInstallCommand,
      }
)

