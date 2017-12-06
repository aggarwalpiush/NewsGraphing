#!/usr/bin/env python
import subprocess as sub
# get mbfc labels

# sub.check_call(["jupyter", "nbconvert", "--to", "notebook","execute", "notebooks/Step 1) Scrape labels from Media Bias Fact Check.ipynb"])
# gets the links and builds the edge list

print("Running mongopull.py")
sub.check_call(["python", "src/mongopull.py", "1000"])

# pulls the text from the mongo and filters out rows that don't have text field
# need to be more flexible/automated
# python src/dataset.py
print("Running content_model_baseline.py")
sub.check_call(["python", "src/content_model_baseline.py"])

# BP Model
sub.check_call(["julia", "src/structure_model_beliefpropagation.jl"])
