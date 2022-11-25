# Copyright 2018-2020 The GPflow Contributors. All Rights Reserved.
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
from typing import Sequence, Tuple

import tensorflow as tf

from check_shapes import Shape, check_shapes
from .inducing_variables import MultioutputInducingVariables, FallbackSharedIndependentInducingVariables, FallbackSeparateIndependentInducingVariables


class MultioutputSpectralInducingVariables(MultioutputInducingVariables):
    
    """
    just a subclass for the time being
    """



class FallbackSharedIndependentSpectralInducingVariables(FallbackSharedIndependentInducingVariables):

    """
    just a subclass for the time being
    """



class FallbackSeparateIndependentSpectralInducingVariables(FallbackSeparateIndependentInducingVariables):

    """
    just a subclass for the time being
    """


class SharedIndependentSpectralInducingVariables(FallbackSharedIndependentSpectralInducingVariables):

    """
    just a subclass for the time being
    """


class SeparateIndependentSpectralInducingVariables(FallbackSeparateIndependentSpectralInducingVariables):

    """
    just a subclass for the time being
    """
