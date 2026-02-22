ENS-GI Digital Twin Documentation
=================================

**Version:** 0.3.0

**One Engine, Three Applications:**

1. **Research Simulator** - Parameter sweeps and bifurcation analysis
2. **Neuromorphic Hardware** - SPICE/Verilog-A export for FPGA/ASIC
3. **Clinical Predictor** - Patient-specific parameterization with AI

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api_reference
   tutorials
   validation_report
   contributing

Overview
--------

The ENS-GI Digital Twin is a multiscale computational model of the enteric nervous system (ENS) and gastrointestinal (GI) motility. It bridges computational neuroscience, hardware engineering, and clinical medicine.

**Key Features:**

* **Biophysically realistic** - Extended Hodgkin-Huxley neurons with 6 ion channels
* **ICC pacemaker** - FitzHugh-Nagumo slow wave generator (~3 cpm)
* **Network dynamics** - Gap junctions + chemical synapses
* **IBS profiles** - IBS-D, IBS-C, IBS-M pathophysiology
* **AI parameter estimation** - PINN + Bayesian inference
* **Virtual drug trials** - 7 FDA-approved drugs with PK/PD models
* **Hardware export** - Complete Verilog-A library (8 modules)

Quick Start
-----------

Installation::

   pip install -r requirements.txt
   python setup.py install

Basic usage::

   from ens_gi_digital.core import ENSGIDigitalTwin

   # Create digital twin
   twin = ENSGIDigitalTwin(n_segments=10)
   twin.apply_profile('healthy')

   # Run simulation
   result = twin.run(duration=2000, dt=0.05)

   # Extract biomarkers
   biomarkers = twin.extract_biomarkers()
   print(twin.clinical_report())

Applications
------------

Research Simulator
~~~~~~~~~~~~~~~~~~

Explore parameter space and analyze network dynamics:

.. code-block:: python

   import numpy as np

   # Parameter sweep
   g_Na_values = np.linspace(80, 160, 10)
   for g_Na in g_Na_values:
       twin = ENSGIDigitalTwin(n_segments=10)
       for neuron in twin.network.neurons:
           neuron.params.g_Na = g_Na
       twin.run(1000, dt=0.1)
       bio = twin.extract_biomarkers()
       print(f"g_Na={g_Na:.1f} → motility={bio['motility_index']:.3f}")

Neuromorphic Hardware
~~~~~~~~~~~~~~~~~~~~~

Export to SPICE/Verilog-A for hardware acceleration:

.. code-block:: python

   twin = ENSGIDigitalTwin(n_segments=20)

   # Export pure SPICE netlist
   twin.export_spice_netlist('network.sp', use_verilog_a=False)

   # Export Verilog-A netlist
   twin.export_spice_netlist('network_va.sp', use_verilog_a=True)

   # Export standalone module
   va_module = twin.export_verilog_a_module()

Clinical Predictor
~~~~~~~~~~~~~~~~~~

Estimate patient-specific parameters from clinical data:

.. code-block:: python

   from ens_gi_digital.pinn import PINNEstimator, PINNConfig
   from ens_gi_digital.bayesian import BayesianEstimator

   # PINN (fast)
   pinn = PINNEstimator(twin, PINNConfig())
   pinn_estimates = pinn.estimate_parameters(
       voltages=egg_signal,
       forces=hrm_signal
   )

   # Bayesian (rigorous)
   bayes = BayesianEstimator(twin)
   trace = bayes.estimate_parameters(egg_signal)
   summary = bayes.summarize_posterior(trace)

Citation
--------

If you use this software in research, please cite:

.. code-block:: bibtex

   @software{ens_gi_digital_twin_2026,
     author = {ENS-GI Digital Twin Contributors},
     title = {ENS-GI Digital Twin: Multiscale Simulation of ENS and GI Motility},
     year = {2026},
     url = {https://github.com/yourusername/ens-gi-digital-twin},
     version = {0.3.0}
   }

License
-------

MIT License. See LICENSE file for details.

Support
-------

* GitHub Issues: https://github.com/yourusername/ens-gi-digital-twin/issues
* Documentation: https://ens-gi-digital-twin.readthedocs.io
* Email: contact@example.com

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
