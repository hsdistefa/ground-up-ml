import pytest


from groundupml.utils import distributions


test_poisson_input = [(1, 0, 0.368),
                      (1, 1, 0.368),
                      (1, 2, 0.184),
                      (1, 3, 0.06131),
                      (1, 4, 0.01533),
                      (1, 5, 0.003066),
                      (2.5, 0, 0.0821),
                      (2.5, 1, 0.2052),
                      (2.5, 2, 0.2565),
                      (0, 0 , 1),
                      (0, 1, 0)
                     ]

@pytest.mark.parametrize("lmda,k,expected", test_poisson_input)
def test_poisson_equal(lmda, k, expected):
    assert distributions.poisson(lmda, k) == pytest.approx(expected, rel=1e-3)

