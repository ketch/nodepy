# Use a non-interactive matplotlib backend for all tests, so that doctests
# and tests that make plots run without a display.
import matplotlib

matplotlib.use('Agg')
