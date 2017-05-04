from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='speedml',
    version='0.8.1',
    description='Speedml Machine Learning Speed Start',
    long_description=long_description,
    url='https://github.com/Speedml/speedml',
    author='Manav Sehgal',
    author_email='new@speedml.com',
    license='MIT',
    keywords='machine-learning data-science jupyter-notebook',
    packages=['speedml'],
    install_requires=[
      'pandas', 'numpy', 'seaborn', 'matplotlib',
      'sklearn', 'xgboost'
    ],
    zip_safe=False,
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 2 - Pre-Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
 )
