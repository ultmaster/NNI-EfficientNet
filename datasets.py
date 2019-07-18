import os
import tensorflow as tf

import config


def is_image(file):
    return file.endswith(".png")


MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def load_and_preprocess_image(path, image_size):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, image_size)
    image /= 255.0  # normalize to [0,1] range

    image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
    return image


def cifar10(meta=False):
    # wget http://pjreddie.com/media/files/cifar.tgz, extract to data/datasets/cifar10
    def prepare_cifar(subset, image_size=None, training=False, batch_size=1):
        folder = os.path.join(rt_path, subset)
        files = list(filter(is_image, os.listdir(folder)))
        labels = [label_id_map[file[file.index("_") + 1:file.index(".")]] for file in files]
        paths = [os.path.join(folder, file) for file in files]
        assert len(paths) == len(labels)
        if meta:
            return {
                "length": len(paths),
                "num_classes": len(label_id_map)
            }

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        def load_and_preprocess_from_path_label(path, label):
            return load_and_preprocess_image(path, image_size), label

        ds = ds.map(load_and_preprocess_from_path_label)
        if training:
            ds = ds.shuffle(8).repeat()
        return ds.batch(batch_size)

    rt_path = config.DATASETS_PATH["cifar10"]
    with open(os.path.join(rt_path, "labels.txt")) as f:
        label_id_map = {word.strip(): i for i, word in enumerate(f.readlines())}

    return prepare_cifar
