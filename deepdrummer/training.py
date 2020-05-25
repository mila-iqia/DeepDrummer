import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal
import soundfile as sf

def load_audio_files(path, single_bar=True):
    """
    Load all wave files in a directory
    Returns a Python array of torch tensors of audio
    """

    audios = []

    for file_root, dirs, files in os.walk(path):
        for name in files:
            # be careful not to get stuck in wrong files like .DS_Store
            if not re.match(r'.*wav', name):
                continue
            name = os.path.join(file_root, name)
            data, sr = sf.read(name)
            assert sr == 44100

            if len(data.shape) == 2 and data.shape[1] == 2:
                data = 0.5 * (data[:, 0] + data[:, 1])

            # We only use the 2nd bar out of 4
            if single_bar:
                if data.shape[0] >= 4*44100:
                    data = data[2*44100:4*44100]
                else:
                    data = data[:2*44100]

            data = data.astype(np.float32)
            data = torch.from_numpy(data).unsqueeze(dim=0)
            audios.append(data)

    return audios

def sample_batch(good_audio, bad_audio, batch_size):
    assert batch_size > 0 and batch_size % 2 == 0
    half_size = batch_size // 2

    # Use smaller minibatches if not enough training data
    half_size = min(good_audio.size(0), half_size)
    half_size = min(bad_audio.size(0), half_size)

    good_idx = np.random.randint(0, good_audio.size(0) - half_size + 1)
    bad_idx = np.random.randint(0, bad_audio.size(0) - half_size + 1)
    batch_good = good_audio[good_idx:(good_idx+half_size)]
    batch_bad = bad_audio[bad_idx:(bad_idx+half_size)]
    batch = torch.cat((batch_good, batch_bad), 0)

    good_labels = torch.ones((half_size, 1), dtype=torch.long)
    bad_labels = torch.zeros((half_size, 1), dtype=torch.long)
    labels = torch.cat((good_labels, bad_labels), 0)

    return batch, labels

def augment(batch, max_shift=128):
    """
    Do data augmentation on a batch of audio data
    """

    # Re-scale the pattern
    scale_dist = Normal(torch.FloatTensor([1]), torch.FloatTensor([0.05]))
    batch = batch * scale_dist.sample().to(batch.device)

    # Move the baseline up/down a bit
    base_dist = Normal(torch.FloatTensor([0]), torch.FloatTensor([0.05]))
    batch = batch + base_dist.sample().to(batch.device)

    # Add some noise
    noise_dist = Normal(torch.FloatTensor([0]), torch.FloatTensor([0.006]))
    noise = noise_dist.sample(sample_shape=batch.size()).to(batch.device)
    noise = noise.view(batch.size())
    batch = batch + noise

    # Shift/rotate the pattern
    shift_amt = np.random.randint(0, max_shift)
    left_slice = batch[:, :, (-shift_amt):]
    right_slice = batch[:, :, :(-shift_amt)]
    batch = torch.cat((left_slice, right_slice), -1)

    return batch

def train_batch(
    model,
    optimizer,
    good_audio,
    bad_audio,
    batch_size,
    device,
    want_augmentation=True,
    want_learning=True,
    want_accuracy=False
):
    """
    Do one iteration of training (one minibatch)
    """

    # Sample a batch
    audio, labels = sample_batch(good_audio, bad_audio, batch_size)
    audio = audio.to(device)
    labels = labels.to(device)

    # Data augmentation
    if want_augmentation:
        audio = augment(audio)

    logits = model(audio)

    loss = F.nll_loss(logits.unsqueeze(-1), labels)

    if want_learning:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = float(loss.detach().cpu().numpy().mean())

    # let's do it like that to avoid breaking current usage of this function
    if want_accuracy:
        labels = labels.reshape([-1])
        preds_labels = logits.max(axis=1)[1] # get the index of the max log-probability
        accuracy = preds_labels.eq(labels).cpu().numpy().mean()
        return (loss, accuracy)
    else:
        return loss
