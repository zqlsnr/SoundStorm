from pathlib import Path

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio


def main():
    # Instantiate a pretrained EnCodec model
    model = EncodecModel.encodec_model_16khz_academic(
        repository=Path("/mnt/Corpus3/temp/chenhaitao/AcademiCodec")
    )
    print(f"Model sample_rate is {model.sample_rate}, channels is {model.channels}")
    # The number of codebooks used will be determined bythe bandwidth selected.
    # E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
    # Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
    # For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
    # of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
    model.set_target_bandwidth(6)

    # Load and pre-process the audio waveform
    for wav in Path("LJSpeech-1.1/").glob("**/*.wav"):
        data, sr = torchaudio.load(wav)
        data = convert_audio(data, sr, model.sample_rate, model.channels)
        data = data.unsqueeze(0)  # [B, c, T]

        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = model.encode(data)
        # print(encoded_frames)
        codes = encoded_frames[0][0]
        print(codes.shape)
        torch.save(codes, wav.with_suffix(".code.pt"))
        # codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
        # break
        """
        aa = model.decode(encoded_frames)
        print(aa.shape)
        print(aa.dtype)
        torchaudio.save("a_16khz.wav", aa[0], 16000, encoding="PCM_S", bits_per_sample=16)
        """


if __name__ == "__main__":
    main()
