# VS-DistilDRBA
**vapoursynth version for [DistilDRBA](https://github.com/routineLife1/DistilDRBA) based on [vs-rife](https://github.com/HolyWu/vs-rife).**

> This project is modified from [HolyWu/vs-rife](https://github.com/HolyWu/vs-rife) and achieves nearly the same interpolation quality as the original [DistilDRBA](https://github.com/routineLife1/DistilDRBA) project.
> 
> With TensorRT integration, it achieves a 400% acceleration, enabling real-time playback on high-performance NVIDIA GPUs.

## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.6.0 or later
- [VapourSynth](http://www.vapoursynth.com/) R66 or later
- [vs-miscfilters-obsolete](https://github.com/vapoursynth/vs-miscfilters-obsolete) (only needed for scene change detection)

`trt` requires additional packages:
- [TensorRT](https://developer.nvidia.com/tensorrt) 10.7.0.post1 or later
- [Torch-TensorRT](https://pytorch.org/TensorRT/) 2.6.0 or later

To install the latest stable version of PyTorch, Torch-TensorRT and cupy, run:
```
pip install -U packaging setuptools wheel
pip install -U torch torchvision torch_tensorrt --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.nvidia.com
pip install -U cupy-cuda12x
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

| model                | scale | os    | hardware           | arch                                                       | fps 720 | fps 1080 | vram 720 | vram 1080 | backend                                                                        | verified output                    | batch | level | streams | threads | onnx      | onnxslim / onnxsim | onnx shape  | trtexec shape | precision | usage                                                                                               |
|----------------------| ----- | ----- |--------------------|------------------------------------------------------------|---------|----------|----------|-----------|--------------------------------------------------------------------------------| ---------------------------------- | ----- | ----- |---------|---------| --------- | ------------------ | ----------- | ------------- | --------- |-----------------------------------------------------------------------------------------------------|
| drba_distilled v1 | 2x    | Linux | rtx5070 / 14600kf | [drba_distilled](https://github.com/routineLife1/DistilDRBA)   | 420     | 120       | 1.7gb    | 3.7gb     | trt 10.8, torch 20241231+cu126, torch_trt 20250102+cu126 (routineLife1 vsdrba) | yes, works                         | 1     | 5     | -       | 8       | -         | -                  | -           | static        | RGBH      | drba_distilled(clip, trt=True, trt_static_shape=True, model="v1", trt_optimization_level=5, sc=False) |
