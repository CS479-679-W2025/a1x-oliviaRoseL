import sys
import os
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import neuron_models as nm
import ans as ans

def test_spike_interpolation():
    # Test interpolated spike time
    net = nm.SpikingNetwork()
    A = ans.LIFNeuron(label='A')
    net.add_neuron(A)

    # Set voltage and current so it's just about to spike.
    A.set_v(0.95)
    A.set_s(10.)

    net.simulate(0.002, 0.001)

    assert A.get_spikes()[0] == pytest.approx(0.00011, abs=1e-5)
    assert A.get_v() == pytest.approx(0., abs=1e-5)
