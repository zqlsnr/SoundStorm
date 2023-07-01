import io
import json
import logging
import time
from pathlib import Path
import torch

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile

from soundstorm_pytorch import ConformerWrapper, SoundStorm

logging.getLogger("numba").setLevel(logging.WARNING)


def load_model(cfg_path, model_path, device):
    cfg = json.load(open(cfg_path))
    conformer = ConformerWrapper(
        codebook_size=cfg["model"]["codebook_size"],
        num_quantizers=cfg["model"]["num_quantizers"],
        conformer=dict(
            dim=cfg["model"]["dim"],
            depth=cfg["model"]["n_layers"],
            heads=cfg["model"]["n_heads"],
            conv_kernel_size=cfg["model"]["kernel_size"],
            attn_flash=False,
        ),
    )
    model = SoundStorm(
        conformer,
        steps=cfg["model"]["steps"],
        schedule="cosine",
        num_semantic_token_ids=500,
        wav2vec_target_sample_hz=16000,
        wav2vec_downsample_factor=320,
        codec_target_sample_hz=16000,
        codec_downsample_factor=320,
    )
    model = model.to(device)
    data = torch.load(model_path, map_location=device)
    model.load_state_dict(data["model"])
    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ns2vc inference")

    # Required
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="logs/model-127.pt",
        help="Path to the model.",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="config.json",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--prompt_code",
        type=str,
        default="config.json",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--prompt_condition",
        type=str,
        default="config.json",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="dataset/zh_ras_M010/000001_hubert.pt",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="dataset/zh_ras_M010/000001_hubert.pt",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-r",
        "--refer_names",
        type=str,
        nargs="+",
        default=["1.wav"],
        help="Reference audio path.",
    )
    parser.add_argument(
        "-n",
        "--clean_names",
        type=str,
        nargs="+",
        default=["2.wav"],
        help="A list of wav file names located in the raw folder.",
    )
    parser.add_argument(
        "-t",
        "--trans",
        type=int,
        nargs="+",
        default=[0],
        help="Pitch adjustment, supports positive and negative (semitone) values.",
    )

    # Optional
    parser.add_argument(
        "-a",
        "--auto_predict_f0",
        action="store_true",
        default=True,
        help="Automatic pitch prediction for voice conversion. Do not enable this when converting songs as it can cause serious pitch issues.",
    )
    parser.add_argument(
        "-cl",
        "--clip",
        type=float,
        default=0,
        help="Voice forced slicing. Set to 0 to turn off(default), duration in seconds.",
    )
    parser.add_argument(
        "-lg",
        "--linear_gradient",
        type=float,
        default=0,
        help="The cross fade length of two audio slices in seconds. If there is a discontinuous voice after forced slicing, you can adjust this value. Otherwise, it is recommended to use. Default 0.",
    )
    parser.add_argument(
        "-fmp",
        "--f0_mean_pooling",
        action="store_true",
        default=False,
        help="Apply mean filter (pooling) to f0, which may improve some hoarse sounds. Enabling this option will reduce inference speed.",
    )

    # generally keep default
    parser.add_argument(
        "-sd",
        "--slice_db",
        type=int,
        default=-40,
        help="Loudness for automatic slicing. For noisy audio it can be set to -30",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda:1",
        help="Device used for inference. None means auto selecting.",
    )
    parser.add_argument(
        "-p",
        "--pad_seconds",
        type=float,
        default=0.5,
        help="Due to unknown reasons, there may be abnormal noise at the beginning and end. It will disappear after padding a short silent segment.",
    )
    parser.add_argument(
        "-wf", "--wav_format", type=str, default="wav", help="output format"
    )
    parser.add_argument(
        "-lgr",
        "--linear_gradient_retain",
        type=float,
        default=0.75,
        help="Proportion of cross length retention, range (0-1]. After forced slicing, the beginning and end of each segment need to be discarded.",
    )
    parser.add_argument(
        "-ft",
        "--f0_filter_threshold",
        type=float,
        default=0.05,
        help="F0 Filtering threshold: This parameter is valid only when f0_mean_pooling is enabled. Values range from 0 to 1. Reducing this value reduces the probability of being out of tune, but increases matte.",
    )

    args = parser.parse_args()

    clean_names = args.clean_names
    refer_names = args.refer_names
    trans = args.trans
    slice_db = args.slice_db
    wav_format = args.wav_format
    auto_predict_f0 = args.auto_predict_f0
    pad_seconds = args.pad_seconds
    clip = args.clip
    lg = args.linear_gradient
    lgr = args.linear_gradient_retain
    F0_mean_pooling = args.f0_mean_pooling
    cr_threshold = args.f0_filter_threshold

    device = args.device
    model = load_model(args.config_path, args.model_path, device)

    c = torch.load(args.condition).unsqueeze(0).to(device)
    a = model.generate_by_level(num_latents=1, cond_semantic_token_ids=c)
    
    print(a)
    torch.save(a.to("cpu"), args.output)


if __name__ == "__main__":
    main()
