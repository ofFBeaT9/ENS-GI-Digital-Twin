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

# Machine learning frameworks — lazy-loaded so that worker processes spawned by
# ProcessPoolExecutor (which import this package to find the simulation worker)
# do not trigger TensorFlow / CUDA initialisation.
# from .pinn import PINNEstimator  ← moved to __getattr__ below

# Drug library
from .drug_library import (
    DrugProfile,
    DrugLibrary,
    VirtualDrugTrial,
    apply_drug,
)

# Data loading
from .patient_data import PatientDataLoader

def __getattr__(name):
    """Lazy-load optional heavy modules to avoid side-effect warnings on import."""
    if name == "PINNEstimator":
        from .pinn import PINNEstimator
        return PINNEstimator
    if name == "BayesianEstimator":
        try:
            from .bayesian import BayesianEstimator
            return BayesianEstimator
        except ImportError:
            return None
    if name == "ClinicalWorkflow":
        try:
            from .clinical_workflow import ClinicalWorkflow
            return ClinicalWorkflow
        except (ImportError, SyntaxError):
            return None
    raise AttributeError(f"module 'ens_gi_digital' has no attribute {name!r}")

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
