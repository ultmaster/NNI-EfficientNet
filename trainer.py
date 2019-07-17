import tensorflow as tf
import numpy as np
from argparse import ArgumentParser

from tensorflow.python.keras.datasets import cifar100, cifar10

from model import get_model_params, build_model


def build_learning_rate(initial_lr,
                        global_step,
                        steps_per_epoch=None,
                        lr_decay_type='exponential',
                        decay_factor=0.97,
                        decay_epochs=2.4,
                        total_steps=None,
                        warmup_epochs=5):
    """Build learning rate."""
    if lr_decay_type == 'exponential':
        assert steps_per_epoch is not None
        decay_steps = steps_per_epoch * decay_epochs
        lr = tf.train.exponential_decay(
            initial_lr, global_step, decay_steps, decay_factor, staircase=True)
    elif lr_decay_type == 'cosine':
        assert total_steps is not None
        lr = 0.5 * initial_lr * (
                1 + tf.cos(np.pi * tf.cast(global_step, tf.float32) / total_steps))
    elif lr_decay_type == 'constant':
        lr = initial_lr
    else:
        assert False, 'Unknown lr_decay_type : %s' % lr_decay_type

    if warmup_epochs:
        tf.logging.info('Learning rate warmup_epochs: %d' % warmup_epochs)
        warmup_steps = int(warmup_epochs * steps_per_epoch)
        warmup_lr = (
                initial_lr * tf.cast(global_step, tf.float32) / tf.cast(
            warmup_steps, tf.float32))
        lr = tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)

    return lr


def build_optimizer(learning_rate,
                    optimizer_name='rmsprop',
                    decay=0.9,
                    epsilon=0.001,
                    momentum=0.9):
    """Build optimizer."""
    if optimizer_name == 'sgd':
        tf.logging.info('Using SGD optimizer')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimizer_name == 'momentum':
        tf.logging.info('Using Momentum optimizer')
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        tf.logging.info('Using RMSProp optimizer')
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay, momentum,
                                              epsilon)
    else:
        raise ValueError('Unknown optimizer:', optimizer_name)

    return optimizer


def get_ema_vars():
    """Get all exponential moving average (ema) variables."""
    ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
    for v in tf.global_variables():
        # We maintain mva for batch norm moving mean and variance as well.
        if 'moving_mean' in v.name or 'moving_variance' in v.name:
            ema_vars.append(v)
    return list(set(ema_vars))


def model_fn(features, labels, mode, params):
    blocks_args, global_params = get_model_params(num_classes=params["num_label_classes"])
    logits, endpoints = build_model(features, blocks_args, global_params, mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    one_hot_labels = tf.one_hot(tf.reshape(labels, (-1, )), params["num_label_classes"])
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=one_hot_labels,
        label_smoothing=params["label_smoothing"])

    # Add weight decay to the loss for non-batch-normalization variables.
    loss = cross_entropy + params["weight_decay"] * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if 'batch_normalization' not in v.name])
    loss = tf.identity(loss, "loss")

    global_step = tf.train.get_global_step()

    if params["moving_average_decay"] > 0:
        ema = tf.train.ExponentialMovingAverage(
            decay=params["moving_average_decay"], num_updates=global_step)
        ema_vars = get_ema_vars()

    # Compute the current epoch and associated learning rate from global_step.
    steps_per_epoch = params["steps_per_epoch"]
    scaled_lr = params["base_learning_rate"] * (params["batch_size"] / 256.0)
    learning_rate = build_learning_rate(scaled_lr, global_step, steps_per_epoch)
    optimizer = build_optimizer(learning_rate)

    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)
    if params["moving_average_decay"] > 0:
        with tf.control_dependencies([train_op]):
            train_op = ema.apply(ema_vars)

    if mode == tf.estimator.ModeKeys.EVAL:
        predictions = tf.argmax(logits, axis=1)
        top_1_accuracy = tf.metrics.accuracy(labels, predictions)

        in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
        top_5_accuracy = tf.metrics.mean(in_top_5)

        metrics = {
            'top_1_accuracy': top_1_accuracy,
            'top_5_accuracy': top_5_accuracy,
        }

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=metrics)

    num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('number of trainable parameters: {}'.format(num_params))

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((tf.image.convert_image_dtype(features, tf.float32), labels))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((tf.image.convert_image_dtype(features, tf.float32), labels))
    return dataset.batch(batch_size)


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.enable_eager_execution()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    params = vars(args)
    tf.logging.info("Running on %d samples" % x_train.shape[0])
    params["steps_per_epoch"] = x_train.shape[0] / args.batch_size
    classifier = tf.estimator.Estimator(model_fn=model_fn, params=params)

    logging_hook = tf.train.LoggingTensorHook(tensors={"loss": "loss"}, every_n_iter=1)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(x_train, y_train, args.batch_size),
                                        hooks=[logging_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(x_test, y_test, args.batch_size))

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base-learning-rate", default=0.256, type=float,
                        help='Base learning rate when train batch size is 256.')
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--weight-decay", default=1e-5, type=float,
                        help='Weight decay coefficiant for l2 regularization.')
    parser.add_argument("--label-smoothing", default=0.1, type=float,
                        help='Label smoothing parameter used in the softmax_cross_entropy')
    parser.add_argument("--moving-average-decay", default=1 - 1e-4, type=float,
                        help='Moving average decay rate')
    parser.add_argument("--num-label-classes", default=1000, type=int)
    main(parser.parse_args())
