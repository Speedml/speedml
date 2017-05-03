import pytest

class TestFeature:

    def test_drop(self, sml):
        message = sml.feature.drop(['Cabin'])
        assert '1' in message

    def test_impute(self, sml):
        message = sml.feature.impute()
        assert '0' in message
