========================
Speedml Python Package
========================

Machine Learning Speed Start
-------------------------------

  Bringing together speed and craft is an awesome experience...

Speedml is a Python package for speed starting machine learning projects.

Install Speedml package using `pip` using the following command::

  pip install speedml

Minimal `method level documentation<http://pythonhosted.org/speedml/>` is available with every new release.

  The `Speedml.com<https://speedml.com>` website documents API use cases, features, best practices, and demos in much more detail.

Speedml is open source and available under MIT license. We manage the project on GitHub.

- `GitHub repo<https://github.com/Speedml/speedml>`
- `Project roadmap<https://github.com/Speedml/speedml/projects/1>`
- `Issues tracking<https://github.com/Speedml/speedml/issues>`
- `Demo notebooks<https://github.com/Speedml/notebooks>`

We are authoring Speedml API with four goals in mind.

Popular. Best packages together
--------------------------------

Speedml already imports and properly initializes the popular ML packages including pandas, numpy, sklearn, xgboost, and matplotlib. All you need to do is import speedml to get started::

  from speedml import Speedml

Rapid. Machine learning speed start
------------------------------------

Coding is up to 3X faster when using Speedml because of (1) iterative development, (2) linear workflow, and (3) component-based API.

These two lines of Speedml code (a) load the training, test datasets, (b) define the target and unique id features, (c) plot the feature correlation matrix heatmap for numerical features::

  sml = Speedml('train.csv', 'test.csv',
                target='Survived', uid='PassengerId')
  sml.plot.correlate()

Easy. Concise commands with sensible defaults
----------------------------------------------

A notebook using Speedml reduces coding required by up to 70%. Speedml API implements methods requiring zero to minimal number of parameters, working on sensible defaults.

Call to this single method replaces empty values in the entire dataframe with median value for numerical features and most common values for text features::

  sml.feature.impute()

Productive. Intuitive linear workflow
---------------------------------------

Learning machine learning fundamentals is a breeze with Speedml as we have designed the API to follow a linear workflow with sensible prerequisites and intuitive next steps.

These five lines of Speedml code perform feature engineering on four features of Titanic dataset::

  sml.feature.fillna(a='Cabin', new='Z')
  sml.feature.regex_extract(new='Deck', a='Cabin', regex='([A-Z]){1}')
  sml.feature.ordinal_to_numeric('Sex', {'male': 0, 'female': 1})
  sml.feature.sum(new='FamilySize', a='Parch', b='SibSp')
  sml.feature.add('FamilySize', 1)

Hope you enjoy using Speedml in your projects. Watch this space as we intend to update Speedml frequently with more cool features.
