from setuptools import setup

setup(
    name='speedml',
    version='0.0.7',
    description='Speedml Rapid Machine Learning Workflow',
    url='http://github.com/manavsehgal/speedml',
    author='Manav Sehgal',
    author_email='new@speedml.com',
    license='MIT',
    packages=['speedml'],
    install_requires=[
      'pandas', 'numpy', 'seaborn', 'matplotlib',
      'sklearn', 'xgboost'
    ],
    zip_safe=False
 )
