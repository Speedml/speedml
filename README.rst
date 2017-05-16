======================================
Speedml Machine Learning Speed Start
======================================

Speedml is a Python package for speed starting machine learning projects.

The speedml.com_ website documents API use cases, behind-the-scenes implementation, features, best practices, and demos in detail.

  To see Speedml in action run or download the notebook_ `Titanic Solution Using Speedml` which walks through an end-to-end machine learning solution documenting features of the Speedml API.

Latest Speedml release is always available on the PyPi_ website.

Install Speedml package using `pip` like so::

  pip install speedml

We manage the project on GitHub.

- Demo notebooks_
- GitHub repo_
- Project roadmap_
- Issues_ tracking

We are authoring Speedml API with four goals in mind.

Popular. Best packages together
--------------------------------

Speedml already imports and properly initializes the popular ML packages including pandas, numpy, sklearn, xgboost, and matplotlib. All you need to do is import speedml to get started::

  from speedml import Speedml

Rapid. Machine learning speed start
------------------------------------

Coding is up to 3X faster when using Speedml because of (1) iterative development, (2) linear workflow, and (3) component-based API.

These three lines of Speedml code (a) load the training, test datasets, (b) define the target and unique id features, (c) plot the feature correlation matrix heatmap for numerical features, (d) perform a detailed EDA returning 10-15 observations and next steps for making the datasets model ready::

  sml = Speedml('train.csv', 'test.csv',
                target='Survived', uid='PassengerId')
  sml.plot.correlate()
  sml.eda()

Easy. Concise commands with sensible defaults
----------------------------------------------

A notebook using Speedml reduces coding required by up to 70%. Speedml API implements methods requiring zero to minimal number of parameters, working on sensible defaults.

Call to this single method replaces empty values in the entire dataframe with median value for numerical features and most common values for text features::

  sml.feature.impute()

Productive. Intuitive linear workflow
---------------------------------------

Understanding machine learning fundamentals is a breeze with Speedml as we have designed the API to follow a linear workflow with sensible prerequisites and intuitive next steps.

These three lines of Speedml code perform feature engineering by replacing null values, extracting a new feature matching a regular expression, and dropping a feature that is no longer required::

  sml.feature.fillna(a='Cabin', new='Z')
  sml.feature.extract(new='Deck', a='Cabin', regex='([A-Z]){1}')
  sml.feature.drop(['Cabin'])

Hope you enjoy using Speedml in your projects. Watch this space as we intend to update Speedml frequently with more cool features.

.. _PyPi: https://pypi.python.org/pypi/speedml
.. _documentation: http://pythonhosted.org/speedml/
.. _speedml.com: https://speedml.com
.. _repo: https://github.com/Speedml/speedml
.. _roadmap: https://github.com/Speedml/speedml/projects/1
.. _notebooks: https://github.com/Speedml/notebooks
.. _Issues: https://github.com/Speedml/speedml/issues
.. _notebook: https://github.com/Speedml/notebooks/blob/master/titanic/titanic-solution-using-speedml.ipynb
