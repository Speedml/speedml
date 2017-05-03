import pytest

@pytest.fixture(scope="module")
def sml():
    from speedml import Speedml
    sml = Speedml(
        train = '/Users/manavsehgal/Developer/speedml/tests/data/train.csv',
        test = '/Users/manavsehgal/Developer/speedml/tests/data/test.csv',
        target = 'Survived',
        uid = 'PassengerId')
    yield sml
    # teardown
    sml = None
