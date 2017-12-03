import pytest

@pytest.fixture(scope="module")
def sml():
    from speedml import Speedml
    sml = Speedml(
        train = 'tests/data/train.csv',
        test = 'tests/data/test.csv',
        target = 'Survived',
        uid = 'PassengerId')
    yield sml
    # teardown
    sml = None
