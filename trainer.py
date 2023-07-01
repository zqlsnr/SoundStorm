import json
import logging
import math
import os
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from random import random

import torch
import torch.nn.functional as F
import torchaudio
from accelerate import Accelerator, DistributedDataParallelKwargs
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from torch import einsum, expm1, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# import utils
from dataset import SoundStormDataset, TextAudioCollate
from soundstorm_pytorch import ConformerWrapper, SoundStorm


def exists(x):
    return x is not None


def clean_checkpoints(path_to_models="logs/44k/", n_ckpts_to_keep=2, sort_by_time=True):
    """Freeing up space by deleting saved ckpts

    Arguments:
    path_to_models    --  Path to the model directory
    n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pt and D_0.pt
    sort_by_time      --  True -> chronologically delete ckpts
                          False -> lexicographically delete ckpts
    """
    ckpts_files = [
        f
        for f in os.listdir(path_to_models)
        if os.path.isfile(os.path.join(path_to_models, f))
    ]
    name_key = lambda _f: int(re.compile("._(\d+)\.pt").match(_f).group(1))
    time_key = lambda _f: os.path.getmtime(os.path.join(path_to_models, _f))
    sort_key = time_key if sort_by_time else name_key
    x_sorted = lambda _x: sorted(
        [f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pt")],
        key=sort_key,
    )
    to_del = [
        os.path.join(path_to_models, fn)
        for fn in (x_sorted("G")[:-n_ckpts_to_keep] + x_sorted("D")[:-n_ckpts_to_keep] + x_sorted("model")[:-n_ckpts_to_keep])
    ]
    del_info = lambda fn: logger.info(f".. Free up space by deleting ckpt {fn}")
    del_routine = lambda x: [os.remove(x), del_info(x)]
    rs = [del_routine(fn) for fn in to_del]


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(
        self,
        cfg_path="./config.json",
        split_batches=True,
    ):
        super().__init__()

        # accelerator

        self.cfg = json.load(open(cfg_path))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs]
            # split_batches = split_batches,
            # mixed_precision = 'bf16' if self.cfg['train']['bf16'] else 'no'
        )
        # print(self.accelerator.device)

        self.accelerator.native_amp = self.cfg["train"]["amp"]
        device = self.accelerator.device

        # model
        conformer = ConformerWrapper(
            codebook_size=self.cfg["model"]["codebook_size"],
            num_quantizers=self.cfg["model"]["num_quantizers"],
            conformer=dict(
                dim=self.cfg["model"]["dim"],
                depth=self.cfg["model"]["n_layers"],
                heads=self.cfg["model"]["n_heads"],
                conv_kernel_size=self.cfg["model"]["kernel_size"],
                attn_flash=False,
            ),
        )
        self.model = SoundStorm(
            conformer,
            steps=self.cfg['model']['steps'],
            schedule="cosine",
            num_semantic_token_ids=500,
            wav2vec_target_sample_hz=16000,
            wav2vec_downsample_factor=320,
            codec_target_sample_hz=16000,
            codec_downsample_factor=320,
        )
        # self.model = SoundStorm(conformer, steps=self.cfg["model"]["steps"], schedule="cosine")
        # print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        self.num_samples = self.cfg["train"]["num_samples"]
        self.save_and_sample_every = self.cfg["train"]["save_and_sample_every"]

        self.batch_size = self.cfg["train"]["train_batch_size"]
        self.gradient_accumulate_every = self.cfg["train"]["gradient_accumulate_every"]

        self.train_num_steps = self.cfg["train"]["train_num_steps"]

        # dataset and dataloader
        collate_fn = TextAudioCollate()
        ds = SoundStormDataset(self.cfg['data']['training_files'])
        self.ds = ds
        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.cfg["train"]["num_workers"],
            collate_fn=collate_fn,
        )

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        
        ds_val = SoundStormDataset(self.cfg['data']['validation_files'])
        self.eval_dl = DataLoader(
            ds_val,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=self.cfg["train"]["num_workers"],
            collate_fn=collate_fn,
        )
        # print(1)
        # optimizer

        self.opt = Adam(
            self.model.parameters(),
            lr=self.cfg["train"]["learning_rate"],
            betas=self.cfg["train"]["adam_betas"],
        )

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(
                self.model,
                beta=self.cfg["train"]["ema_decay"],
                update_every=self.cfg["train"]["ema_update_every"],
            )
            self.ema.to(self.device)

        self.logs_folder = Path(self.cfg["train"]["logs_folder"])
        self.logs_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler)
            else None,
        }

        torch.save(data, str(self.logs_folder / f"model_{milestone}.pt"))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(
            str(self.logs_folder / f"model-{milestone}.pt"), map_location=device
        )

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train(self):
        # print(1)
        accelerator = self.accelerator
        device = accelerator.device

        if accelerator.is_main_process:
            logger = get_logger(self.cfg["train"]["logs_folder"])
            writer = SummaryWriter(log_dir=self.cfg["train"]["logs_folder"])
            writer_eval = SummaryWriter(
                log_dir=os.path.join(self.cfg["train"]["logs_folder"], "eval")
            )

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:
            while self.step < self.train_num_steps:
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    data = [d.to(device) for d in data]
                    c, codes = data
                    # print(len(data))

                    with self.accelerator.autocast():
                        loss, loss_parts = self.model(codes, cond_semantic_token_ids=c)
                        loss = loss / self.gradient_accumulate_every

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(
                    f"loss: {loss:.4f}, generator_loss: {loss_parts.generator_loss}, critic_loss: {loss_parts.critic_loss}"
                )

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()
                # print(loss_diff, loss_f0, ce_loss)
                ############################logging#############################################
                if accelerator.is_main_process and self.step % self.cfg["train"]["log_interval"] == 0:
                    logger.info(
                        "Train step: {}, Epoch: {} [{:.0f}%], Losses: {}".format(
                            self.step,
                            (self.step * self.batch_size) // len(self.ds),
                            100.0 * self.step / self.train_num_steps,
                            loss,
                        )
                    )

                    scalar_dict = {
                        "loss/all": loss,
                    }

                    summarize(
                        writer=writer,
                        global_step=self.step,
                        scalars=scalar_dict,
                    )

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        (
                            c_padded,
                            codes_padded,
                        ) = next(iter(self.eval_dl))
                        c, codes = (
                            c_padded.to(device),
                            codes_padded.to(device),
                        )
                        #print(c.shape)
                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_samples_list = list(
                                map(
                                    lambda n: self.ema.ema_model.generate_by_level(
                                        num_latents=1,  # 实际未使用
                                        cond_semantic_token_ids=c,
                                        batch_size=n,
                                    ),
                                    batches,
                                )
                            )

                        #print(len(all_samples_list))
                        #print(all_samples_list)
                        #print(all_samples_list[0].shape)

                        all_samples = torch.cat(all_samples_list, dim=0).detach().cpu()
                        # print(all_samples)
                        # print(all_samples.shape)
                        torch.save(all_samples, f"{self.cfg['train']['logs_folder']}/test_{milestone}.pt")

                        """
                        torchaudio.save(
                            str(self.logs_folder / f"sample-{milestone}.wav"),
                            all_samples,
                            24000,
                        )
                        
                        audio_dict = {}
                        audio_dict.update(
                            {f"gen/audio": all_samples, f"gt/audio": wav_padded[0]}
                        )
                        summarize(
                            writer=writer_eval,
                            global_step=self.step,
                            audios=audio_dict,
                            audio_sampling_rate=24000,
                        )
                        """
                        keep_ckpts = self.cfg["train"]["keep_ckpts"]
                        if keep_ckpts > 0:
                            clean_checkpoints(
                                path_to_models=self.cfg["train"]["logs_folder"],
                                n_ckpts_to_keep=keep_ckpts,
                                sort_by_time=True,
                            )
                        self.save(milestone)

                pbar.update(1)

        accelerator.print("training complete")
