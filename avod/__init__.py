"""avod: profile-conditioned video–odor prediction."""

from .backbone import ProfileBackbone, ThreeWayUserGating
from .baselines import (
    AVOnlyBaseline,
    AVNaiveUserBaseline,
    ProfileModel,
    MMCLIPBaseline,
    UniformProfileBaseline,
)

__all__ = [
    "ProfileBackbone",
    "ThreeWayUserGating",
    "AVOnlyBaseline",
    "AVNaiveUserBaseline",
    "ProfileModel",
    "MMCLIPBaseline",
    "UniformProfileBaseline",
]

__version__ = "0.1.0"
