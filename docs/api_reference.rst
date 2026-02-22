API Reference
=============

Core Modules
------------

ens_gi_digital.core
~~~~~~~~~~~~~~~~~~~

.. automodule:: ens_gi_digital.core
   :members:
   :undoc-members:
   :show-inheritance:

ENSGIDigitalTwin
^^^^^^^^^^^^^^^^

.. autoclass:: ens_gi_digital.core.ENSGIDigitalTwin
   :members:
   :special-members: __init__

ENSNeuron
^^^^^^^^^

.. autoclass:: ens_gi_digital.core.ENSNeuron
   :members:
   :special-members: __init__

ICCPacemaker
^^^^^^^^^^^^

.. autoclass:: ens_gi_digital.core.ICCPacemaker
   :members:
   :special-members: __init__

SmoothMuscle
^^^^^^^^^^^^

.. autoclass:: ens_gi_digital.core.SmoothMuscle
   :members:
   :special-members: __init__

ENSNetwork
^^^^^^^^^^

.. autoclass:: ens_gi_digital.core.ENSNetwork
   :members:
   :special-members: __init__

AI Modules
----------

ens_gi_digital.pinn
~~~~~~~~~~~~~~~~~~~

.. automodule:: ens_gi_digital.pinn
   :members:
   :undoc-members:
   :show-inheritance:

PINNEstimator
^^^^^^^^^^^^^

.. autoclass:: ens_gi_digital.pinn.PINNEstimator
   :members:
   :special-members: __init__

PINNConfig
^^^^^^^^^^

.. autoclass:: ens_gi_digital.pinn.PINNConfig
   :members:

ens_gi_digital.bayesian
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ens_gi_digital.bayesian
   :members:
   :undoc-members:
   :show-inheritance:

BayesianEstimator
^^^^^^^^^^^^^^^^^

.. autoclass:: ens_gi_digital.bayesian.BayesianEstimator
   :members:
   :special-members: __init__

BayesianConfig
^^^^^^^^^^^^^^

.. autoclass:: ens_gi_digital.bayesian.BayesianConfig
   :members:

PriorSpec
^^^^^^^^^

.. autoclass:: ens_gi_digital.bayesian.PriorSpec
   :members:

Drug Library
------------

ens_gi_digital.drug_library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ens_gi_digital.drug_library
   :members:
   :undoc-members:
   :show-inheritance:

DrugLibrary
^^^^^^^^^^^

.. autoclass:: ens_gi_digital.drug_library.DrugLibrary
   :members:

DrugProfile
^^^^^^^^^^^

.. autoclass:: ens_gi_digital.drug_library.DrugProfile
   :members:

VirtualDrugTrial
^^^^^^^^^^^^^^^^

.. autoclass:: ens_gi_digital.drug_library.VirtualDrugTrial
   :members:
   :special-members: __init__

Functions
---------

Parameter Estimation
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ens_gi_digital.pinn.build_mlp_network
.. autofunction:: ens_gi_digital.pinn.build_resnet_network
.. autofunction:: ens_gi_digital.bayesian.get_default_priors

Drug Utilities
~~~~~~~~~~~~~~

.. autofunction:: ens_gi_digital.drug_library.apply_drug
.. autofunction:: ens_gi_digital.drug_library.compute_plasma_concentration
.. autofunction:: ens_gi_digital.drug_library.apply_drug_effect

Constants and Enums
-------------------

IBS_PROFILES
~~~~~~~~~~~~

Dictionary of IBS profile configurations:

- **healthy**: Normal gut function
- **ibs_d**: Diarrhea-predominant IBS (hyperexcitable)
- **ibs_c**: Constipation-predominant IBS (hypoexcitable)
- **ibs_m**: Mixed IBS (variable dynamics)

TargetType
~~~~~~~~~~

.. autoclass:: ens_gi_digital.drug_library.TargetType
   :members:
   :undoc-members:
