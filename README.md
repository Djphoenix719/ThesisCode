# An Empirical Study on the Efficacy of Evolutionary Algorithms for Automated Neural Architecture Engineering
## Andrew Cuccinello
Contact: `cuccinela5@students.rowan` (rowan is an educational domain, which I have not added here to avoid spam)

### Install
1. You'll need to grab a copy of FFMPEG to save videos of Pong. Place `ffmpeg.exe`, `ffplay.exe`, and `ffprobe.exe` in the root directory. On Unix based systems you'll need different files.
2. Download the NASBench dataset(s)
   1. The partial dataset
      > curl -O https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
   2. (and/or) The full dataset
      > curl -O https://storage.googleapis.com/nasbench/nasbench_full.tfrecord
3. Create two environments. I used Anaconda for one and a venv for another, as some of the dependencies of packages conflict. I have provided the `environment.yml` and `requirements.txt` files.
   1. Install NASBench into the root directory:
      > https://github.com/google-research/nasbench
   2. You will need to ensure you use the right version of tensorflow with NASBench, it only supports v1.x, so you may have to build it from old source like I did.
   3. ALE is a pain to install on Windows, but there are a number of articles describing varying ways to set it up which can be found via Google. You can build it yourself, use WSL, or use a Windows compatible version like [this one](https://github.com/Kojoley/atari-py).
4. Ensure you edit any paths while you get an idea of how the code works. A lot of them are hardcoded (e.g. where saved models get output to), and any settings you want to change (e.g. number of children, size of the population, etc.).

### Run
#### Pong benchmarks
```shell
   python -m benchmarks
```
#### NASBench trials
```shell
   python -m trials
```
#### Generate video
```shell
   python video.py
```
