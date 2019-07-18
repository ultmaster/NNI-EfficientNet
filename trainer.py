import logging
import os
from datetime import datetime

import tensorflow as tf
import numpy as np
from argparse import ArgumentParser

from datasets import cifar10
from exporter import NNIExporter
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
    blocks_args, global_params = get_model_params(num_classes=params["num_label_classes"],
                                                  width_coefficient=params["width_coefficient"],
                                                  depth_coefficient=params["depth_coefficient"])
    model = build_model(blocks_args, global_params)

    if params["data_format"] == "channels_first":
        features = tf.transpose(features, [0, 3, 1, 2])

    with tf.variable_scope("efficient-net-logits"):
        logits = model(features, training=mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.identity(logits, 'logits')

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
        logits=logits, onehot_labels=one_hot_labels,
        label_smoothing=params["label_smoothing"])

    # Add weight decay to the loss for non-batch-normalization variables.
    loss = cross_entropy
    if params["weight_decay"]:
        loss += params["weight_decay"] * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()
             if 'batch_normalization' not in v.name])
    loss = tf.identity(loss, "train_loss")

    global_step = tf.train.get_global_step()

    if params["moving_average_decay"] > 0:
        ema = tf.train.ExponentialMovingAverage(
            decay=params["moving_average_decay"], num_updates=global_step)
        ema_vars = get_ema_vars()

    steps_per_epoch = params["steps_per_epoch"]
    scaled_lr = params["base_learning_rate"] * (params["batch_size"] / 256.0)
    learning_rate = build_learning_rate(scaled_lr, global_step, steps_per_epoch)
    optimizer = build_optimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)
    if params["moving_average_decay"] > 0:
        with tf.control_dependencies([train_op]):
            train_op = ema.apply(ema_vars)

    predictions = tf.argmax(logits, axis=1)
    top_1_accuracy = tf.metrics.accuracy(labels, predictions)

    if mode == tf.estimator.ModeKeys.EVAL:
        in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
        top_5_accuracy = tf.metrics.mean(in_top_5)

        metrics = {
            'top_1_accuracy': top_1_accuracy,
            'top_5_accuracy': top_5_accuracy,
        }

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=metrics)
    else:
        tf.summary.scalar("train_acc", top_1_accuracy[1])

    num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('number of trainable parameters: {}'.format(num_params))

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    image_size = [args.resolution, args.resolution]
    if args.dataset == "cifar10":
        cifar_prepare, dataset_gen = cifar10(True), cifar10(False)
        train_meta, test_meta = cifar_prepare("train"), cifar_prepare("test")
    else:
        raise NotImplementedError

    params = vars(args)
    tf.logging.info("Training on %d samples, evaluation on %d samples" % (train_meta["length"],
                                                                          test_meta["length"]))
    params["steps_per_epoch"] = train_meta["length"] // args.batch_size
    max_steps = args.num_epochs * params["steps_per_epoch"]
    params["num_label_classes"] = train_meta["num_classes"]

    if args.request_from_nni:
        exporters = [NNIExporter()]
    else:
        exporters = []

    # TODO: add multi-gpu support
    # if os.environ.get("CUDA_VISIBLE_DEVICES"):
    #     strategy = tf.contrib.distribute.MirroredStrategy()
    # else:
    strategy = None
    run_config = tf.estimator.RunConfig(model_dir=args.log_dir,
                                        log_step_count_steps=10,
                                        save_checkpoints_secs=args.evaluation_interval,
                                        save_summary_steps=10,
                                        train_distribute=strategy,
                                        eval_distribute=strategy)
    classifier = tf.estimator.Estimator(model_fn=model_fn, params=params, config=run_config)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: dataset_gen("train", image_size,
                                                                     True, args.batch_size),
                                        max_steps=max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: dataset_gen("test", image_size,
                                                                   False, args.batch_size),
                                      throttle_secs=args.evaluation_interval,
                                      exporters=exporters)

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


if __name__ == "__main__":
    logger = logging.getLogger('efficientnet')

    parser = ArgumentParser()
    parser.add_argument("--depth-coefficient", default=1.0, type=float)
    parser.add_argument("--width-coefficient", default=1.0, type=float)
    parser.add_argument("--resolution", default=224, type=int)
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"], type=str)
    parser.add_argument("--log-dir", default="logs", type=str)
    parser.add_argument("--base-learning-rate", default=0.256, type=float,
                        help='Base learning rate when train batch size is 256.')
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--weight-decay", default=1e-5, type=float,
                        help='Weight decay coefficiant for l2 regularization.')
    parser.add_argument("--label-smoothing", default=0.1, type=float,
                        help='Label smoothing parameter used in the softmax_cross_entropy')
    parser.add_argument("--moving-average-decay", default=1 - 1e-4, type=float,
                        help='Moving average decay rate')
    parser.add_argument("--data-format", choices=["channels_first", "channels_last"], default="channels_last",
                        help="Prefer channels first on GPU, otherwise choose channels last")
    parser.add_argument("--num-epochs", default=5, type=int, help="Number of epochs in total")
    parser.add_argument("--evaluation-interval", default=180, type=int,
                        help="Frequency of evaluation (and report to NNI)")
    parser.add_argument("--request-from-nni", default=False, action="store_true")

    args = parser.parse_args()
    if args.request_from_nni:
        import nni
        tuner_params = nni.get_next_parameter()
        tf.logging.info(tuner_params)

        parser.depth_coefficient = tuner_params["alpha"]
        parser.width_coefficient = tuner_params["beta"]
        parser.resolution = int(tuner_params["gamma"] * 224)

        args.log_dir = os.environ["NNI_OUTPUT_DIR"]
        tf.logging.info(args)

    try:
        main(args)
    except Exception as exception:
        tf.logging.error(exception)
        raise
