# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model Builder for EfficientNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tensorflow as tf

import efficientnet_model


def build_model(images,
                model_name,
                training,
                override_params=None,
                model_dir=None):
    """A helper functiion to creates a model and returns predicted logits.

    Args:
      images: input images tensor.
      model_name: string, the predefined model name.
      training: boolean, whether the model is constructed for training.
      override_params: A dictionary of params for overriding. Fields must exist in
        efficientnet_model.GlobalParams.
      model_dir: string, optional model dir for saving configs.

    Returns:
      logits: the logits tensor of classes.
      endpoints: the endpoints for each layer.

    Raises:
      When model_name specified an undefined model, raises NotImplementedError.
      When override_params has invalid fields, raises ValueError.
    """
    assert isinstance(images, tf.Tensor)
    blocks_args, global_params = get_model_params(model_name, override_params)

    if model_dir:
        param_file = os.path.join(model_dir, 'model_params.txt')
        if not tf.gfile.Exists(param_file):
            if not tf.gfile.Exists(model_dir):
                tf.gfile.MakeDirs(model_dir)
            with tf.gfile.GFile(param_file, 'w') as f:
                tf.logging.info('writing to %s' % param_file)
                f.write('model_name= %s\n\n' % model_name)
                f.write('global_params= %s\n\n' % str(global_params))
                f.write('blocks_args= %s\n\n' % str(blocks_args))

    with tf.variable_scope(model_name):
        model = efficientnet_model.Model(blocks_args, global_params)
        logits = model(images, training=training)

    logits = tf.identity(logits, 'logits')
    return logits, model.endpoints


def build_model_base(images, model_name, training, override_params=None):
    """A helper functiion to create a base model and return global_pool.

    Args:
      images: input images tensor.
      model_name: string, the model name of a pre-defined MnasNet.
      training: boolean, whether the model is constructed for training.
      override_params: A dictionary of params for overriding. Fields must exist in
        mnasnet_model.GlobalParams.

    Returns:
      features: global pool features.
      endpoints: the endpoints for each layer.

    Raises:
      When model_name specified an undefined model, raises NotImplementedError.
      When override_params has invalid fields, raises ValueError.
    """
    assert isinstance(images, tf.Tensor)
    blocks_args, global_params = get_model_params(model_name, override_params)

    with tf.variable_scope(model_name):
        model = efficientnet_model.Model(blocks_args, global_params)
        features = model(images, training=training, features_only=True)

    features = tf.identity(features, 'global_pool')
    return features, model.endpoints
