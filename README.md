# News Graphing

This repository contains code for studying the network of fake news, including
code and data for _JF, NF, NK, and EB_ "Credibility in the News: Do we need to read?"
WSDM 2018 workshop on Misinformation and Misbehavior online, http://jpfairbanks.com/mis2-2018.

This codebase includes both a python package and a collection of julia scripts
for analyzing fake news. We use NLP tools in python including the fabulous SpaCy
library, and classifiers in sklearn. The Belief Propagation that appears here is
written in Julia using LightGraphs.jl graphs.

We use PostgreSQL and MongoDB to store structured and unstructured data regarding
the articles. For the purpose of reproducibility a static snapshot of the data
has been made available through the `src/newgraphing/download.py` script. In our
live system this analysis can be conducted against the data as it is updated
every 15 minutes.

## Getting Started

Here are some instructions to get going with this project

### Depedencies

* julia
* python, pip
* git-lfs

### Installing

1. Download the code `git clone github.com/jpfairbanks/newsgraphing && cd newsgraphing`
2. Install all dependencies with `pip install .`
3. Install all julia dependencies in the `src/fakeprop/REQUIRE` file with `Pkg.add`
3. Run the main script to generate figures `python ./main.py`

## Contributing

If anything is unclear or doesn't work, let us know on the issues page. Feel
free to ask for help by opening a new issue and we will take a look and answer
your questions.

### Contributors

* James Fairbanks
* Nate Knauf
* Natalie Fitch
* David Ediger
* Erica Briscoe
