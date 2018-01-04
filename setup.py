from setuptools import setup, find_packages

package = 'newsgraphing'
version = '0.1'
from setuptools import setup
from setuptools.command.install import install
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
      packages=['src'],
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
      package_dir={'bsdetector': 'bsdetector'},
      description="Analyzes news articles for bias and credibility",
      url='url',
      cmdclass={
	      # 'develop': PostDevelopCommand,
	      'install': PostInstallCommand,
      }
)


