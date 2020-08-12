# DeepDrummer

DeepDrummer is a drum machine that uses a deep learning model to generate drum loops. The model is trained interactively based on user ratings (like/dislike), and tries to approximate user preferences. It is a tool for quickly exploring new musical ideas.

The accompanying [DeepDrummer paper](https://arxiv.org/abs/2008.04391) is available on arXiV.org.

## Installation

Dependencies:
- Python 3.6+
- PyTorch 1.0+
- torchaudio
- soundfile
- sounddevice
- numpy
- matplotlib
- scipy
- six
- tkinter

Installing the Python 3 package:

```
git clone git@github.com:mila-iqia/DeepDrummer.git
cd DeepDrummer
pip3 install -e .
```

Running the standalone GUI application:
```
python3 -m deepdrummer.standalone
```
