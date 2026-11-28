# Use a non-interactive matplotlib backend for all tests, so that doctests
# and tests that make plots run without a display.
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all figures after each test (doctests included), so that one
    test's plots cannot affect another's."""
    yield
    plt.close('all')
