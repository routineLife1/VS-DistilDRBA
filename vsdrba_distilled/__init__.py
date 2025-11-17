from __future__ import annotations

import math
import os
import sys
import warnings
from fractions import Fraction
from threading import Lock
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vapoursynth as vs
from .__main__ import download_model
from torch._decomp import get_decompositions

__version__ = "5.4.1"

os.environ["CI_BUILD"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

warnings.filterwarnings("ignore", "Both operands of the binary elementwise op")
warnings.filterwarnings("ignore", "The given NumPy array is not writable")

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

models = [
    "v1",
    "v2_lite"
]

@contextmanager
def redirect_stdout_to_stderr():
    old_stdout = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.close(old_stdout)


@redirect_stdout_to_stderr()
@torch.inference_mode()
def drba_distilled(
        clip: vs.VideoNode,
        device_index: int = 0,
        model: str = "v1",
        auto_download: bool = False,
        factor_num: int = 2,
        factor_den: int = 1,
        fps_num: int | None = None,
        fps_den: int | None = None,
        scale: float = 1.0,
        trt: bool = False,
        trt_debug: bool = False,
        trt_workspace_size: int = 0,
        trt_max_aux_streams: int | None = None,
        trt_optimization_level: int | None = None,
        trt_cache_dir: str = model_dir,
) -> vs.VideoNode:
    """Real-Time Intermediate Flow Estimation for Video Frame Interpolation

    :param clip:                    Clip to process. Only RGBH and RGBS formats are supported.
                                    RGBH performs inference in FP16 mode while RGBS performs inference in FP32 mode.
    :param device_index:            Device ordinal of the GPU.
    :param model:                   Model to use.
    :param auto_download:           Automatically download the specified model if the file has not been downloaded.
    :param factor_num:              Numerator of factor for target frame rate.
    :param factor_den:              Denominator of factor for target frame rate.
                                    For example `factor_num=5, factor_den=2` will multiply the frame rate by 2.5.
    :param fps_num:                 Numerator of target frame rate.
    :param fps_den:                 Denominator of target frame rate.
                                    Override `factor_num` and `factor_den` if specified.
    :param scale:                   Control the process resolution for optical flow model. Try scale=0.5 for 4K video.
                                    Must be 0.25, 0.5, 1.0, 2.0, or 4.0.
                                    Leave the argument as None if the frames already have _SceneChangeNext property set.
    :param trt:                     Use TensorRT for high-performance inference.
                                    Not supported in '4.0' and '4.1' models.
    :param trt_debug:               Print out verbose debugging information.
    :param trt_workspace_size:      Size constraints of workspace memory pool.
    :param trt_max_aux_streams:     Maximum number of auxiliary streams per inference stream that TRT is allowed to use
                                    to run kernels in parallel if the network contains ops that can run in parallel,
                                    with the cost of more memory usage. Set this to 0 for optimal memory usage.
                                    (default = using heuristics)
    :param trt_optimization_level:  Builder optimization level. Higher level allows TensorRT to spend more building time
                                    for more optimization options. Valid values include integers from 0 to the maximum
                                    optimization level, which is currently 5. (default is 3)
    :param trt_cache_dir:           Directory for TensorRT engine file. Engine will be cached when it's built for the
                                    first time. Note each engine is created for specific settings such as model
                                    path/name, precision, workspace etc, and specific GPUs and it's not portable.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("drba: this is not a clip")

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("drba: only RGBH and RGBS formats are supported")

    if clip.num_frames < 2:
        raise vs.Error("drba: clip's number of frames must be at least 2")

    if not torch.cuda.is_available():
        raise vs.Error("drba: CUDA is not available")

    if model not in models:
        raise vs.Error(f"drba: model must be one of {models}")

    if factor_num < 1:
        raise vs.Error("drba: factor_num must be at least 1")

    if factor_den < 1:
        raise vs.Error("drba: factor_den must be at least 1")

    if fps_num is not None and fps_num < 1:
        raise vs.Error("drba: fps_num must be at least 1")

    if fps_den is not None and fps_den < 1:
        raise vs.Error("drba: fps_den must be at least 1")

    if fps_num is not None and fps_den is not None and clip.fps == 0:
        raise vs.Error("drba: clip does not have a valid frame rate and hence fps_num and fps_den cannot be used")

    if scale not in [0.25, 0.5, 1.0, 2.0, 4.0]:
        raise vs.Error("drba: scale must be 0.25, 0.5, 1.0, 2.0, or 4.0")

    if not os.path.exists(os.path.join(model_dir, f"{model}.pkl")) or os.path.getsize(os.path.join(model_dir, f"{model}.pkl")) == 0:
        if auto_download:
            download_model(f"https://github.com/routineLife1/VS-DistilDRBA/releases/download/model/{model}.pkl")
        else:
            raise vs.Error(
                "drba: model file has not been downloaded. run `python -m vsdrba_distilled` to download all models, or set "
                "`auto_download=True` to only download the specified model"
            )

    torch.set_float32_matmul_precision("high")

    fp16 = clip.format.bits_per_sample == 16
    dtype = torch.half if fp16 else torch.float

    device = torch.device("cuda", device_index)

    modulo = 64
    match model:
        case "v1":
            from .distilDRBA import IFNet

        case "v2_lite":
            from .distilDRBA_v2_lite import IFNet


    model_name = f"{model}.pkl"

    if fps_num is not None and fps_den is not None:
        factor = Fraction(fps_num, fps_den) / clip.fps
        factor_num, factor_den = factor.as_integer_ratio()

    if factor_num <= factor_den:
        raise vs.Error(
            "drba: target frame rate must be higher than source frame rate. consider using change_fps from "
            "https://github.com/Jaded-Encoding-Thaumaturgy/vs-tools if you want to reduce the frame rate"
        )

    w = clip.width
    h = clip.height
    tmp = max(modulo, int(modulo / scale))
    pw = math.ceil(w / tmp) * tmp
    ph = math.ceil(h / tmp) * tmp
    padding = (0, pw - w, 0, ph - h)
    need_pad = any(p > 0 for p in padding)

    if trt:
        import tensorrt
        import torch_tensorrt

        dimensions = f"{pw}x{ph}"

        flownet_engine_path = os.path.join(
            os.path.realpath(trt_cache_dir),
            (
                    f"DistilDRBA_{model_name}"
                    + f"_{dimensions}"
                    + f"_{'fp16' if fp16 else 'fp32'}"
                    + f"_scale-{scale}"
                    + f"_{torch.cuda.get_device_name(device)}"
                    + f"_trt-{tensorrt.__version__}"
                    + (f"_workspace-{trt_workspace_size}" if trt_workspace_size > 0 else "")
                    + (f"_aux-{trt_max_aux_streams}" if trt_max_aux_streams is not None else "")
                    + (f"_level-{trt_optimization_level}" if trt_optimization_level is not None else "")
                    + ".ts"
            ),
        )

        if not os.path.isfile(flownet_engine_path):
            from torch_tensorrt.dynamo.conversion.impl.grid import GridSamplerInterpolationMode

            GridSamplerInterpolationMode.update(
                {
                    0: tensorrt.InterpolationMode.LINEAR,
                    1: tensorrt.InterpolationMode.NEAREST,
                    2: tensorrt.InterpolationMode.CUBIC,
                }
            )

            if sys.stdout is None:
                sys.stdout = open(os.devnull, "w")

            flownet = init_module(model_name, IFNet, scale, device, dtype)

            flownet_inputs = (
                torch.zeros([1, 3, ph, pw], dtype=dtype, device=device),
                torch.zeros([1, 3, ph, pw], dtype=dtype, device=device),
                torch.zeros([1, 3, ph, pw], dtype=dtype, device=device),
                torch.zeros([1], dtype=dtype, device=device),
            )

            flownet_program = torch.export.export(flownet, flownet_inputs)

            flownet = torch_tensorrt.dynamo.compile(
                flownet_program,
                flownet_inputs,
                device=device,
                debug=trt_debug,
                num_avg_timing_iters=4,
                workspace_size=trt_workspace_size,
                min_block_size=1,
                max_aux_streams=trt_max_aux_streams,
                optimization_level=trt_optimization_level,
                use_explicit_typing=True,
            )

            torch_tensorrt.save(flownet, flownet_engine_path, output_format="torchscript", inputs=flownet_inputs)

        flownet = torch.jit.load(flownet_engine_path).eval()

    else:
        flownet = init_module(model_name, IFNet, scale, device, dtype)

    inf_stream = torch.cuda.Stream(device)
    inf_f2t_stream = torch.cuda.Stream(device)
    inf_t2f_stream = torch.cuda.Stream(device)

    inf_stream_lock = Lock()
    inf_f2t_stream_lock = Lock()
    inf_t2f_stream_lock = Lock()

    torch.cuda.current_stream(device).synchronize()

    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        with inf_f2t_stream_lock, torch.cuda.stream(inf_f2t_stream):
            # t = n * factor_den % factor_num / factor_num
            t = 0.5 + n * factor_den % factor_num / factor_num

            if t == 1:
                return f[1]

            img0 = frame_to_tensor(f[0], device)
            img1 = frame_to_tensor(f[1], device)
            img2 = frame_to_tensor(f[2], device)
            if need_pad:
                img0 = F.pad(img0, padding)
                img1 = F.pad(img1, padding)
                img2 = F.pad(img2, padding)

            inf_f2t_stream.synchronize()

        with inf_stream_lock, torch.cuda.stream(inf_stream):
            output = flownet(img0, img1, img2, torch.tensor([t], device=img0.device, dtype=img0.dtype))

            inf_stream.synchronize()

        with inf_t2f_stream_lock, torch.cuda.stream(inf_t2f_stream):
            if need_pad:
                output = output[:, :, :h, :w]

            return tensor_to_frame(output, f[0].copy(), inf_t2f_stream)

    # --------------------------------------------------------
    # 新时间区间 t ∈ [0.5, 1.5] → 滑动窗口整体右移一个 frame
    # --------------------------------------------------------

    # clip0 = frame[i+1]
    clip0 = clip.std.DuplicateFrames(clip.num_frames - 1)[1:]
    clip0 = vs.core.std.Interleave([clip0] * factor_num)

    # clip1 = frame[i+2]
    clip1 = clip.std.DuplicateFrames(clip.num_frames - 1)[2:]
    clip1 = vs.core.std.Interleave([clip1] * factor_num)

    # clip2 = frame[i+3]
    clip2 = clip.std.DuplicateFrames(clip.num_frames - 1).std.DuplicateFrames(clip.num_frames - 1)[3:]
    clip2 = vs.core.std.Interleave([clip2] * factor_num)

    # clip0 = vs.core.std.Interleave([clip] * factor_num)
    # clip1 = clip.std.DuplicateFrames(clip.num_frames - 1)[1:]
    # clip1 = vs.core.std.Interleave([clip1] * factor_num)
    # clip2 = clip.std.DuplicateFrames(clip.num_frames - 1).std.DuplicateFrames(clip.num_frames - 1)[2:]
    # clip2 = vs.core.std.Interleave([clip2] * factor_num)

    if factor_den > 1:
        clip0 = clip0[::factor_den]
        clip1 = clip1[::factor_den]
        clip2 = clip2[::factor_den]

    return clip0.std.FrameEval(lambda n: clip0.std.ModifyFrame([clip0, clip1, clip2], inference),
                               clip_src=[clip0, clip1, clip2])


def init_module(
        model_name: str,
        IFNet: nn.Module,
        scale: float,
        device: torch.device,
        dtype: torch.dtype,
) -> nn.Module:
    state_dict = torch.load(os.path.join(model_dir, model_name), map_location="cpu", weights_only=True, mmap=True)

    with torch.device("meta"):
        flownet = IFNet(scale)
    flownet.load_state_dict(state_dict, strict=True, assign=True)
    flownet.eval().to(device, dtype)

    return flownet


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(np.asarray(frame[plane])).to(device, non_blocking=True)
            for plane in range(frame.format.num_planes)
        ]
    ).unsqueeze(0)


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame, stream: torch.cuda.Stream) -> vs.VideoFrame:
    tensor = tensor.squeeze(0).detach()
    tensors = [tensor[plane].to("cpu", non_blocking=True) for plane in range(frame.format.num_planes)]

    stream.synchronize()

    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), tensors[plane].numpy())
    return frame
