import pytest


from groundupml.utils import distributions


def test_poisson():
    lmda = 1
    k = 0
    assert distributions.poisson(lmda, k) == pytest.approx(0.368, rel=1e-3)

