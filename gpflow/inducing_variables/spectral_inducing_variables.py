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
from ..kernels import MultipleSpectralBlock

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

class RectangularSpectralInducingPointsBase(SpectralInducingVariables):
    def __init__(self, 
        kern: MultipleSpectralBlock,
        name: Optional[str] = None
        ):
        """
        :param kern: contains all the information needed to parametrize 
        :param means: #TODO -- document param and expected shape as well
        :param bandwidths: #TODO -- document param and expected shape as well
        :param variances:#TODO -- focument param and expected shape as well
        """

        super().__init__(name=name)
        self.kern = kern

        """
        #TODO -- I think this might cause some problems with shape_check in posteriors.py
        if not isinstance(means, (tf.Variable, tfp.util.TransformedVariable)):
            means = Parameter(means, transform=positive())
        self.means = means

        if not isinstance(bandwidths, (tf.Variable, tfp.util.TransformedVariable)):
            bandwidths = Parameter(bandwidths, transform=positive())
        self.bandwidths = bandwidths

        if not isinstance(variances, (tf.Variable, tfp.util.TransformedVariable)):
            variances = Parameter(variances, transform=positive())
        self.variances = variances
        """

    @property  # type: ignore[misc]  # mypy doesn't like decorated properties.
    @check_shapes(
        "return: []",
    )
    def num_inducing(self) -> Optional[tf.Tensor]:
        return self.kern.n_components

    @property
    def shape(self) -> Shape:
        shape = self.kern.means.shape
        if not shape:
            return None
        return tuple(shape) + (1,)


class RectangularSpectralInducingPoints(RectangularSpectralInducingPointsBase):
    """
    Real-space (in output space) "spectral" inducing points with a PSD given by symmetrical rectangles. Corresponding kernel is the sinc kernel.
    """
