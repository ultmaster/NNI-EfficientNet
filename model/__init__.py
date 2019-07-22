import json
import os

import tensorflow as tf

import config
from model.block_decoder import BlockDecoder
from model.efficientnet import EfficientNetModel
from model.params import GlobalParams


def get_model_params(width_coefficient=None,
                     depth_coefficient=None,
                     image_size=None,
                     dropout_rate=0.2,
                     drop_connect_rate=0.2,
                     num_classes=1000):
    """Creates a efficientnet model."""
    with open(config.EFFICIENT_NET_BLOCKS_PATH, "r") as f:
        blocks_args = json.load(f)
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        data_format='channels_last',
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        image_size=image_size,
        depth_divisor=8,
        min_depth=None,
        relu_fn=tf.nn.swish)
    decoder = BlockDecoder()
    return decoder.decode(blocks_args), global_params


def get_pretrained_model_params(model_name, override_params=None):
    """Get the block args and global params for a given model."""
    if model_name.startswith('efficientnet'):
        with open(config.EFFICIENT_NET_PRETRAINED_PATH, "r") as f:
            params_dict = json.load(f)
        width_coefficient, depth_coefficient, image_size, dropout_rate = params_dict[model_name]
        blocks_args, global_params = get_model_params(
            width_coefficient, depth_coefficient, image_size, dropout_rate)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)

    if override_params:
        # ValueError will be raised here if override_params has fields not included
        # in global_params.
        global_params = global_params._replace(**override_params)

    tf.logging.info('global_params= %s', global_params)
    tf.logging.info('blocks_args= %s', blocks_args)
    return blocks_args, global_params


def build_model(blocks_args, global_params, model_dir=None):
    """A helper functiion to creates a model and returns predicted logits.

    Args:
      blocks_args:
      global_params: see params.py for details
      model_dir: optional directory to save the model

    Returns:
      logits: the logits tensor of classes.
      endpoints: the endpoints for each layer.

    Raises:
      When model_name specified an undefined model, raises NotImplementedError.
      When override_params has invalid fields, raises ValueError.
    """

    if model_dir:
        param_file = os.path.join(model_dir, 'model_params.txt')
        if not tf.gfile.Exists(param_file):
            if not tf.gfile.Exists(model_dir):
                tf.gfile.MakeDirs(model_dir)
            with tf.gfile.GFile(param_file, 'w') as f:
                tf.logging.info('writing to %s' % param_file)
                f.write('global_params = %s\n\n' % str(global_params))
                f.write('blocks_args = %s\n\n' % str(blocks_args))

    model = EfficientNetModel(blocks_args, global_params)

    return model
