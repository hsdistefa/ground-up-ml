import pytest


from groundupml.utils import functions


test_entropy_input = [([1,0], 1),
                      ([0,0], 0),
                      ([0,1,0], 0.9183),
                      ([-1,1,1,1,-1], 0.971),
                      ([5,-6,5,0,1,1], 1.918),
                      ([1,1,2,2,3,3,4,4,5,5], 2.322)
                     ]

test_gini_input = [([1,0], 0.5),
                   ([0,0], 0),
                   ([0,1,0], 0.444444),
                   ([-1,1,1,1,-1], 0.48),
                   ([5,-6,5,0,1,1], 0.722222),
                   ([1,1,2,2,3,3,4,4,5,5], 0.8)
                  ]

test_info_gain_input = [([[1,0],[0,1]], 0),
                        ([[1,1,1,1,1,1,1,1,0,0],[1,1,0,0,0,0,0,0,0,0]], 0.278),
                        ([[1,1,1,1,1,1,1,1,0,0,0,0,0,0],[1,1,0,0,0,0]], 0.03487),
                        ([[1,1,2,3,5,5],[2,3,4,4]], 0.5712)
                       ]

@pytest.mark.parametrize("y,expected", test_entropy_input)
def test_entropy_equal(y, expected):
    assert functions.entropy(y) == pytest.approx(expected, rel=1e-3)
    
@pytest.mark.parametrize("y,expected", test_gini_input)
def test_gini_equal(y, expected):
    assert functions.gini(y) == pytest.approx(expected, rel=1e-3)

@pytest.mark.parametrize("y_splits,expected", test_info_gain_input)
def test_info_gain_equal(y_splits, expected):
    assert functions.information_gain(y_splits, 'entropy') == \
        pytest.approx(expected, rel=1e-3)