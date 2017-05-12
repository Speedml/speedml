import pytest
import random

class TestFeature:
    # The tests execute in order of definition on the same dataset.
    # If first test drops a feature, second test does not have that feature.
    # Define the test_methods in order of typical notebook workflow.
    # TODO: What is the order of executin of test_classes?

    def test_drop(self, sml):
        message = sml.feature.drop(['Cabin'])
        assert 'Dropped 1 feature' in message

    def test_impute(self, sml):
        message = sml.feature.impute()
        # The return message represents before/after check
        assert 'empty values to 0' in message

    def test_mapping(self, sml):
        # Get a random sample
        idx = random.randrange(0, sml.train.shape[0] - 1)

        # Get value of Sex for idx, convert to numeric
        before = sml.train.get_value(idx, 'Sex')
        before = 0 if before == 'male' else 1

        # Convert all values of Sex to numeric
        sml.feature.mapping('Sex', { 'male': 0, 'female': 1 })

        # Get value of Sex for idx
        after = sml.train.get_value(idx, 'Sex')

        # compare both values
        assert before == after

    def test_fillna(self, sml):
        a = 'Embarked'
        before = sml.train[a].isnull().sum() + sml.test[a].isnull().sum()

        message = sml.feature.fillna('Embarked', 'S')

        assert str(before) in message
