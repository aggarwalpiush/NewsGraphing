#!/usr/bin/env python
import subprocess as sub
import os.path
# get mbfc labels

# sub.check_call(["jupyter", "nbconvert",
# "--to", "notebook","execute", "notebooks/Step 1) Scrape labels from Media Bias Fact Check.ipynb"])
# gets the links and builds the edge list

assert os.path.exists('test'), 'testing directory is not present are you running from project root?'

print("Running mongopull.py")
tmpfile = 'test/tmp.csv'
sub.check_call(["python", "src/mongopull.py", "1000"])
sub.check_call(["python", "src/mongopull.py", "100", '-q', '-o', tmpfile])
if not os.path.exists(tmpfile):
    raise(AssertionError('could not find file {}'.format(tmpfile)))
else:
    print("cleaning up mongopull.py by removing {}".format(tmpfile))
    os.remove(tmpfile)

# pulls the text from the mongo and filters out rows that don't have text field
# need to be more flexible/automated
# python src/dataset.py
print("Running content_model_baseline.py")
sub.check_call(["python", "src/content_model_baseline.py"])

# BP Model
sub.check_call(["julia", "src/structure_model_beliefpropagation.jl"])
