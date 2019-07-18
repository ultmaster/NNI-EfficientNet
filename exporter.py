import tensorflow as tf


class NNIExporter(tf.estimator.Exporter):
    def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
        print(eval_result)
        print(export_path)
        print(checkpoint_path)

    @property
    def name(self):
        return "nni_exporter"
