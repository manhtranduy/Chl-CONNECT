import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pytest

from common.classification_functions import classif5


def test_classif5_invalid_shape():
    # Prepare input with wrong number of spectral bands (should be 6 for OLCI)
    bad_input = np.ones((4, 5))
    with pytest.raises((AssertionError, ValueError)):
        classif5(bad_input)
