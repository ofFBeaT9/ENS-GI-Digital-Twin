"""
ENS-GI Digital Twin - Enteric Nervous System & Gastrointestinal Simulation

A physics-informed digital twin for simulating gut motility and the enteric
nervous system, with machine learning frameworks for patient-specific
parameter estimation.
"""

__version__ = "0.3.0"

# Core simulation
from .core import (
    ENSGIDigitalTwin,
    ENSNeuron,
    ENSNetwork,
    ICCPacemaker,
    SmoothMuscle,
    IBS_PROFILES,
)

# Machine learning frameworks
from .pinn import PINNEstimator

try:
    from .bayesian import BayesianEstimator
except ImportError:
    # PyMC3 is optional
    BayesianEstimator = None

# Drug library
from .drug_library import (
    DrugProfile,
    DrugLibrary,
    VirtualDrugTrial,
    apply_drug,
)

# Data loading
from .patient_data import PatientDataLoader

# Clinical workflow
try:
    from .clinical_workflow import ClinicalWorkflow
except (ImportError, SyntaxError):
    # Skip if has syntax issues
    ClinicalWorkflow = None

__all__ = [
    "ENSGIDigitalTwin",
    "ENSNeuron",
    "ENSNetwork",
    "ICCPacemaker",
    "SmoothMuscle",
    "IBS_PROFILES",
    "PINNEstimator",
    "BayesianEstimator",
    "DrugProfile",
    "DrugLibrary",
    "VirtualDrugTrial",
    "apply_drug",
    "PatientDataLoader",
    "ClinicalWorkflow",
]
