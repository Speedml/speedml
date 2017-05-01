# Speedml | Machine Learning Speed Starter | Python Package

Speedml is a simple and powerful API to speed start your Machine Learning projects. It integrates best ML libraries and popular strategies used by top data scientists in an easy to use Python package.

Get started by installing the Python package.

```
pip install speedml
```

Then all you need to do is include one package in your Jupyter Notebook project to get started.

```
from speedml import Speedml
```

Now you can add your train and test datasets with one single command.

```
sml = Speedml(train = 'data/train.csv',
              test = 'data/test.csv',
              target = 'Survived',
              uid = 'PassengerId')
```

Speedml enhances the sklearn API so you can reuse your knowledge.

```
sml.shape()
```

Results in enhanced shape information.

```
Shape: train (891, 11) test (418, 10)
Numerical: train_n (6) test_n (5)
```

You can directly access popular libraries like pandas.

```
sml.train.head()
```

Using single commands you can perform multi-line Python operations.

```
sml.plot.correlate()
```

Please watch this space for more information about Speedml API including demos, sample Notebooks, and more code.
