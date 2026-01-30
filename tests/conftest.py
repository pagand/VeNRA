import os
import sys

# Add src to python path so pytest can find the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
