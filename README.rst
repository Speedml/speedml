========================
Speedml Python Package
========================

Machine Learning Speed Start
-------------------------------

Speedml is a Python package to speed start machine learning projects. It integrates best ML packages and popular strategies used by top data scientists in an easy to use Python package.

Get started by installing the Python package::

  pip install speedml

Then all you need to do is include one package in your Jupyter Notebook project to get started::

  from speedml import Speedml

Now you can configure the project datasets with one single command::

  sml = Speedml('data/train.csv', 'data/test.csv',
                target = 'Survived', uid = 'PassengerId')

Speedml enhances popular API so the results and side effects are intuitive::

  sml.shape()

Results in enhanced shape information like so::

  Shape: train (891, 11) test (418, 10)
  Numerical: train_n (6) test_n (5)

You can directly access popular libraries like pandas::

  sml.train.head()

Using single commands you can perform multi-line Python operations::

  # Plot feature correlation graph
  sml.plot.correlate()

  # Plot subplots with feature distributions
  sml.plot.distribute()

  # Replace missing values with numerical median or most common text
  sml.feature.impute()

  # Perform feature selection based on model assigned importance
  sml.plot.importance()

Please watch this space for more information about Speedml API including demos, sample Notebooks, and more code.
