import tensorflow as tf
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu_function


def drop_connect(inputs, is_training, drop_connect_rate):
    """Apply drop connect."""
    if not is_training:
        return inputs

    # Compute keep_prob
    # TODO(tanmingxing): add support for training progress.
    keep_prob = 1.0 - drop_connect_rate

    # Compute drop_connect tensor
    batch_size = tf.shape(inputs)[0]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.div(inputs, keep_prob) * binary_tensor
    return output


class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D, tf.layers.Layer):
    """Wrap keras DepthwiseConv2D to tf.layers."""

    pass


class TpuBatchNormalization(tf.layers.BatchNormalization):
    # class TpuBatchNormalization(tf.layers.BatchNormalization):
    """Cross replica batch normalization."""

    def __init__(self, fused=False, **kwargs):
        if fused in (True, None):
            raise ValueError('TpuBatchNormalization does not support fused=True.')
        super(TpuBatchNormalization, self).__init__(fused=fused, **kwargs)

    def _cross_replica_average(self, t, num_shards_per_group):
        """Calculates the average value of input tensor across TPU replicas."""
        num_shards = tpu_function.get_tpu_context().number_of_shards
        group_assignment = None
        if num_shards_per_group > 1:
            if num_shards % num_shards_per_group != 0:
                raise ValueError('num_shards: %d mod shards_per_group: %d, should be 0'
                                 % (num_shards, num_shards_per_group))
            num_groups = num_shards // num_shards_per_group
            group_assignment = [[
                x for x in range(num_shards) if x // num_shards_per_group == y
            ] for y in range(num_groups)]
        return tpu_ops.cross_replica_sum(t, group_assignment) / tf.cast(
            num_shards_per_group, t.dtype)

    def _moments(self, inputs, reduction_axes, keep_dims):
        """Compute the mean and variance: it overrides the original _moments."""
        shard_mean, shard_variance = super(TpuBatchNormalization, self)._moments(
            inputs, reduction_axes, keep_dims=keep_dims)

        num_shards = tpu_function.get_tpu_context().number_of_shards or 1
        if num_shards <= 8:  # Skip cross_replica for 2x2 or smaller slices.
            num_shards_per_group = 1
        else:
            num_shards_per_group = max(8, num_shards // 8)
        tf.logging.info('TpuBatchNormalization with num_shards_per_group %s',
                        num_shards_per_group)
        if num_shards_per_group > 1:
            # Compute variance using: Var[X]= E[X^2] - E[X]^2.
            shard_square_of_mean = tf.math.square(shard_mean)
            shard_mean_of_square = shard_variance + shard_square_of_mean
            group_mean = self._cross_replica_average(
                shard_mean, num_shards_per_group)
            group_mean_of_square = self._cross_replica_average(
                shard_mean_of_square, num_shards_per_group)
            group_variance = group_mean_of_square - tf.math.square(group_mean)
            return (group_mean, group_variance)
        else:
            return (shard_mean, shard_variance)
