# Trained models
We provide trained models on CULane, Tusimple, and CurveLanes.

| Dataset    | Backbone | F1   | Link |
|------------|----------|-------|------|
| CULane     | ResNet18 | 75.0  |  [Google](https://drive.google.com/file/d/1oEjJraFr-3lxhX_OXduAGFWalWa6Xh3W/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1Z3W4y3eA9xrXJ51-voK4WQ?pwd=pdzs)    |
| CULane     | ResNet34 | 76.0  |   [Google](https://drive.google.com/file/d/1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1PHNpVHboQlmpjM5NXl9IxQ?pwd=jw8f)   |
| Tusimple   | ResNet18 | 96.11 |   [Google](https://drive.google.com/file/d/1Clnj9-dLz81S3wXiYtlkc4HVusCb978t/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1umHo0RZIAQ1l_FzL2aZomw?pwd=6xs1)   |
| Tusimple   | ResNet34 | 96.24 |   [Google](https://drive.google.com/file/d/1pkz8homK433z39uStGK3ZWkDXrnBAMmX/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1Eq7oxnDoE0vcQGzs1VsGZQ?pwd=b88p)   |
| CurveLanes | ResNet18 | 80.42 |   [Google](https://drive.google.com/file/d/1VfbUvorKKMG4tUePNbLYPp63axgd-8BX/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1jCqKqgSQdh6nwC5pYpYO1A?pwd=urhe)   |
| CurveLanes | ResNet34 | 81.34 |   [Google](https://drive.google.com/file/d/1O1kPSr85Icl2JbwV3RBlxWZYhLEHo8EN/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1fk2Wg-1QoHXTnTlasSM6uQ?pwd=4mn3)   |


# Choose this model
| Dataset    | Backbone | F1   | Link |
|------------|----------|-------|------|
| CULane     | ResNet34 | 76.0  |  [Google](https://drive.google.com/file/d/1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1PHNpVHboQlmpjM5NXl9IxQ?pwd=jw8f)   |


# Tensorrt Deploy
We also provide a python script to do tensorrt inference on videos.

1. Convert to onnx model
    ```
    python deploy/pt2onnx.py --config_path configs/culane_res34.py --model_path weights/culane_res34.pth
    ```
    Or you can download the onnx model using the following script: https://github.com/PINTO0309/PINTO_model_zoo/blob/main/324_Ultra-Fast-Lane-Detection-v2/download.sh. And copy `ufldv2_culane_res34_320x1600.onnx` to `weights/ufldv2_culane_res34_320x1600.onnx`

2. Convert to tensorrt model

    Use trtexec to convert engine model

    `trtexec --onnx=weights/culane_res34.onnx --saveEngine=weights/culane_res34.engine`

3. Do inference
    ```
    python deploy/trt_infer.py --config_path  configs/culane_res34.py --engine_path weights/culane_res34.engine --video_path example.mp4
    ```