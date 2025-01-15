import sys
import os
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import neuron_models as nm
import ans as ans

def test_step():
    A = ans.LIFNeuron(Tau_m=0.03, label='A')
    A.set_input_buffer(2.1)
    A.set_v(-0.9)
    A.set_s(1.8)
    A.slope()
    A.step(0.02)

    assert A._LIFNeuron__v == pytest.approx(0.9, abs=1e-4)
    assert A._LIFNeuron__s == pytest.approx(43.08, abs=1e-4)
