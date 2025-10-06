# AI Guard

## Setting up the environment

```bash
sudo apt update
sudo apt install -y pulseaudio pulseaudio-utils libasound2 libasound2-plugins portaudio19-dev python3-pyaudio alsa-utils
```

Force ALSA to use system wide plugins

```bash
export ALSA_PLUGIN_DIR=/usr/lib/x86_64-linux-gnu/alsa-lib
python guard_activation.py
```

```bash
conda update -n base -c defaults conda
```
