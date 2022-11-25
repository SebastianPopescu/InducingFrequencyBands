from .inducing_variables import (
    FallbackSeparateIndependentInducingVariables,
    FallbackSharedIndependentInducingVariables,
    MultioutputInducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)

from .spectral_inducing_variables import (
    FallbackSeparateIndependentSpectralInducingVariables,
    FallbackSharedIndependentSpectralInducingVariables,
    MultioutputSpectralInducingVariables,
    SeparateIndependentSpectralInducingVariables,
    SharedIndependentSpectralInducingVariables,
)

__all__ = [
    "FallbackSeparateIndependentInducingVariables",
    "FallbackSharedIndependentInducingVariables",
    "MultioutputInducingVariables",
    "SeparateIndependentInducingVariables",
    "SharedIndependentInducingVariables",
    "inducing_variables",
    "FallbackSeparateIndependentSpectralInducingVariables",
    "FallbackSharedIndependentSpectralInducingVariables",
    "MultioutputSpectralInducingVariables",
    "SeparateIndependentSpectralInducingVariables",
    "SharedIndependentSpectralInducingVariables",
    "spectral_inducing_variables"
]
