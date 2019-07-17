# TPU Version of EfficientNet

http://githuib.com/tensorflow/tpu/tree/master/models/official/efficientnet

Evaluation with b0:

```
python tpu/eval_ckpt_main.py --model_name=efficientnet-b0 --ckpt_dir=pretrained/efficientnet-b0 --example_img=data/examples/panda.jpg --labels_map_file=data/examples/labels_map.txt
```
