import tensorflow as tf


class NNIExporter(tf.estimator.Exporter):
    def export(self, estimator, export_path, checkpoint_path, eval_result,
               is_the_final_export):
        import nni
        result = eval_result["top_1_accuracy"]
        if is_the_final_export:
            nni.report_intermediate_result(result)
        else:
            nni.report_intermediate_result(result)

    @property
    def name(self):
        return "nni_exporter"
