# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp
from check_shapes import Shape, check_shapes
from deprecated import deprecated

from ..base import Module, Parameter, TensorData, TensorType
from ..utilities import positive
from ..kernels import Kernel
from ..kernels.spectral_utils import matern_spectral_density

class SpectralInducingVariables(Module, abc.ABC):
    """
    Abstract base class for inducing variables residing in the spectral domain.
    """

    @property
    @abc.abstractmethod
    def num_inducing(self) -> tf.Tensor:
        """
        Returns the number of inducing variables, relevant for example to determine the size of the
        variational distribution.
        """
        raise NotImplementedError

    @deprecated(
        reason="len(iv) should return an `int`, but this actually returns a `tf.Tensor`."
        " Use `iv.num_inducing` instead."
    )
    def __len__(self) -> tf.Tensor:
        return self.num_inducing

    @property
    @abc.abstractmethod
    def shape(self) -> Shape:
        """
        Return the shape of these inducing variables.

        Shape should be some variation of ``[M, D, P]``, where:

        * ``M`` is the number of inducing variables.
        * ``D`` is the number of input dimensions.
        * ``P`` is the number of output dimensions (1 if this is not a multi-output inducing
          variable).
        """

class SpectralInducingPointsBase(SpectralInducingVariables):
    def __init__(self, 
        a,
        b,
        omegas, 
        name: Optional[str] = None
        ):
        """
        :param omegas: #TODO -- document it 
        :param a: #TODO -- document param and expected shape as well
        :param b: #TODO -- document param and expected shape as well
        """

        super().__init__(name=name)
        self.a = a
        self.b = b
        self.omegas = omegas

    @property  # type: ignore[misc]  # mypy doesn't like decorated properties.
    @check_shapes(
        "return: []",
    )
    def num_inducing(self) -> Optional[tf.Tensor]:
        return tf.shape(self.omegas)[0] + tf.shape(self.omegas[self.omegas != 0])[0]

    @property
    def shape(self) -> Shape:
        shape = self.omegas.shape #FIXME -- not sure this is right, might not cause any problems downstream though
        if not shape:
            return None
        return tuple(shape) + (1,)

    def spectrum(self, kernel):
        
        return matern_spectral_density(self.omegas, kernel)


class SpectralInducingPoints(SpectralInducingPointsBase):
    """
    Real-space (in output space) "spectral" inducing points with a PSD given by 
    the spectrum of the kernel used, currently supporting just the Matern1/2.
    """



