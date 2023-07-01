# SoundStorm
Implementation of SoundStorm, Efficient Parallel Audio Generation from Google Deepmind, in Pytorch.

## Demo
see demos directory, train dataset is 3 hours private chinese dataset.

## Input
use Encodec\_16k\_320 from  <a href="https://github.com/yangdongchao/AcademiCodec">AcademiCodec</a>.
If you want use the script soundstream.py in dataset directory, you can install <a href="https://github.com/chenht2010/encodec">encodec</a>. And download the Encodec\_16k\_32 model, rename it to encodec_16khz_320d-ff892d09.th, put it to where you want, and change the soundstream.py

use <a href="https://github.com/TencentGameMate/chinese_speech_pretrain">chinese_hubert</a>

## Appreciation
Thanks to <a href="https://github.com/lucidrains/soundstorm-pytorch">soundstorm-pytorch</a>, <a href="https://github.com/adelacvg/NS2VC">NS2VC</a>
