# VS-DistilDRBA
**vapoursynth version for [DistilDRBA](https://github.com/routineLife1/DistilDRBA) based on [vs-rife](https://github.com/HolyWu/vs-rife).**

> This project is modified from [HolyWu/vs-rife](https://github.com/HolyWu/vs-rife) and achieves nearly the same interpolation quality as the original [DistilDRBA](https://github.com/routineLife1/DistilDRBA) project.
> 
> With TensorRT integration, it achieves a 400% acceleration, enabling real-time playback on high-performance NVIDIA GPUs.

## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.6.0 or later
- [VapourSynth](http://www.vapoursynth.com/) R66 or later

`trt` requires additional packages:
- [TensorRT](https://developer.nvidia.com/tensorrt) 10.7.0.post1 or later
- [Torch-TensorRT](https://pytorch.org/TensorRT/) 2.6.0 or later

To install the latest stable version of PyTorch, Torch-TensorRT and cupy, run:
```
pip install -U packaging setuptools wheel
pip install -U torch torchvision torch_tensorrt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.nvidia.com
```

## Installation
```
pip install -U vsdrba_distilled==1.0.0
```
If you want to download all models at once, run `python -m vsdrba_distilled`. If you prefer to only download the model you
specified at first run, set `auto_download=True` in `drba_distilled()`.

## Usage
```python
from vsdrba_distilled import drba_distilled
ret = drba_distilled(clip)
```

See `__init__.py` for the description of the parameters.


## Benchmarks

| model                  | scale | os    | hardware           | arch                                                       | speed(fps) 720 | speed(fps) 1080 | vram 720 | vram 1080 | backend         | verified output                    | batch | level | streams | threads| trtexec shape | precision | usage                                                                                               |
|------------------------| ----- | ----- |--------------------|------------------------------------------------------------|----------------|-----------------|----------|-----------|-----------------| ---------------------------------- | ----- |-------|---------|-------| ------------- | --------- |-----------------------------------------------------------------------------------------------------|
| drba_distilled v1      | 2x    | Linux | rtx5070 / 14600kf | [drba_distilled](https://github.com/routineLife1/DistilDRBA)   | 251            | 115             | 1.8gb    | 2.9gb     | torch+trt cu128 | yes, works                         | 1     | 5     | -       | 1 | static        | RGBH      | drba_distilled(clip, trt=True, model="v1", trt_optimization_level=5) |
| drba_distilled v2_lite | 2x    | Linux | rtx5070 / 14600kf | [drba_distilled](https://github.com/routineLife1/DistilDRBA)   | 999+           | 700             | -        | -         | torch+trt cu128 | yes, works                         | 1     | 5     | -       | 1 | static        | RGBH      | drba_distilled(clip, trt=True, model="v1", trt_optimization_level=5) |

## 🤗 Acknowledgement
This project is supported by [SVFI](https://doc.svfi.group/) Development Team.
