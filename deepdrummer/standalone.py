"""
Standalone TkInter GUI app

Run with:
python3 -m deepdrummer.standalone
"""

import re
import os
import time
import math
import argparse
import tkinter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal
import sounddevice as sd
import soundfile as sf
from .patterns import *
from .models import *
from .training import *

class DeepDrummer:
    """
    Encapsulate the DeepDrummer app
    """

    def __init__(self, args, human_eval=False):
        self.args = args
        self.device = torch.device(args.device)

        self.model = MFCCCritic(
            n_conv=args.num_conv,
            n_mlp=args.num_mlp,
            p_dropout=args.dropout
        )
        self.model.to(self.device)
        print('model size:', self.model.num_params)

        # Optimizer for the critic model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        self.good_patterns = []
        self.bad_patterns = []
        self.good_audio = []
        self.bad_audio = []
        self.good_audio_torch = None
        self.bad_audio_torch = None
        self.cur_pattern = None
        self.cur_audio = None
        self.running_loss = None

        self.model_path = os.path.join(args.save_path, 'model.pt')
        self.good_path = os.path.join(args.save_path, 'good')
        self.bad_path = os.path.join(args.save_path, 'bad')

        # Load an saved model if one already exists
        if os.path.exists(self.model_path):
            print('Loading model from "{}"'.format(self.model_path))
            self.model.load_state_dict(torch.load(self.model_path))

        # Load saved audio data if available
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
            os.mkdir(self.good_path)
            os.mkdir(self.bad_path)
        else:
            self.good_audio = load_audio_files(self.good_path)
            self.bad_audio = load_audio_files(self.bad_path)
            if len(self.good_audio) > 0:
                self.good_audio_torch = torch.stack(self.good_audio)
            if len(self.bad_audio) > 0:
                self.bad_audio_torch = torch.stack(self.bad_audio)

        self.init_ui(human_eval)

    def init_ui(self, human_eval):
        """
        Initialize the user interface
        """

        self.window = tkinter.Tk()
        self.window.title("DeepDrummer")
        self.window.resizable(False, False)

        # Variable for the 4-on-the-floor checkbox (enabled by default)
        self.fotf_var = tkinter.IntVar(value=1)

        # Create all the UI elements
        generate_btn = tkinter.Button(self.window, text="Generate")
        like_btn = tkinter.Button(self.window, text="Like :)", bg='#0C0', activebackground='#0C0')
        dislike_btn = tkinter.Button(self.window, text="Dislike :(", bg='#C00', activebackground='#C00')
        self.score_label = tkinter.Label(self.window, text="", width=6, relief="ridge")
        self.good_label = tkinter.Label(self.window, text="0", width=6, relief="ridge")
        self.bad_label = tkinter.Label(self.window, text="0", width=6, relief="ridge")
        random_btn = tkinter.Button(self.window, text="Random")
        sample_good_btn = tkinter.Button(self.window, text="Sample Good")
        sample_bad_btn = tkinter.Button(self.window, text="Sample Bad")
        fotf_check = tkinter.Checkbutton(self.window, text="Enforce 4 on the floor", variable=self.fotf_var)

        # Place the core UI elements on the grid
        generate_btn.grid(column=0, row=0, sticky='nsew', padx=5, pady=5)
        like_btn.grid(column=0, row=1, sticky='nsew', padx=5, pady=5)
        dislike_btn.grid(column=0, row=2, sticky='nsew', padx=5, pady=5)

        # Bind the buttons to callbacks
        generate_btn.bind("<Button-1>", lambda e: self.generate())
        sample_good_btn.bind("<Button-1>", lambda e: self.sample_good())
        sample_bad_btn.bind("<Button-1>", lambda e: self.sample_bad())
        random_btn.bind("<Button-1>", lambda e: self.sample_random())
        like_btn.bind("<Button-1>", lambda e: self.save_good())
        dislike_btn.bind("<Button-1>", lambda e: self.save_bad())

        self.window.bind('<space>', lambda e: self.generate())

        # If in human evaluation mode
        if human_eval:
            # In human eval mode, disable the close button
            self.window.protocol("WM_DELETE_WINDOW", lambda: None)

            # TODO: place count of patterns left to sample

        else:
            # In normal mode, escape closes the window
            self.window.bind('<Escape>', lambda e: self.exit())

            # Place the UI elements on the grid
            self.score_label.grid(column=1, row=0, sticky='nsew', padx=5, pady=5)
            self.good_label.grid(column=1, row=1, sticky='nsew', padx=5, pady=5)
            self.bad_label.grid(column=1, row=2, sticky='nsew', padx=5, pady=5)
            random_btn.grid(column=2, row=0, sticky='nsew', padx=5, pady=5)
            sample_good_btn.grid(column=2, row=1, sticky='nsew', padx=5, pady=5)
            sample_bad_btn.grid(column=2, row=2, sticky='nsew', padx=5, pady=5)
            fotf_check.grid(row=3, columnspan=3, sticky='nsew', padx=5, pady=5)

            # Show count of good and bad audio instances
            self.good_label.configure(text=str(len(self.good_audio)))
            self.bad_label.configure(text=str(len(self.bad_audio)))

    def event_loop(self):
        # Start the training loop
        self.train_loop()

        # Start the event loop
        self.window.mainloop()

    def exit(self):
        print('Quitting program')
        sd.stop()
        self.window.destroy()

    @property
    def fotf_checked(self):
        return self.fotf_var.get() != 0

    def generate(self):
        """
        Sample from a distribution using the Metropolis-Hastings algorithm
        """

        sd.stop()
        pattern, audio, p = sample_metro(self.model, self.fotf_checked)
        self.play_pattern(pattern, audio, p)

    def sample_good(self):
        """
        Try to generate a good pattern using hillclimbing
        """

        sd.stop()
        pattern, audio, p = sample_metro(self.model, self.fotf_checked, min_p=0.95)
        self.play_pattern(pattern, audio, p)

    def sample_bad(self):
        """
        Try to generate a bad pattern using hillclimbing
        """

        sd.stop()
        pattern, audio, p = hillclimb(self.model, fotf=self.fotf_checked, target_score=0, target_dist=0.05)
        self.play_pattern(pattern, audio, p)

    def sample_random(self):
        """
        Generate a completely random pattern
        """

        sd.stop()
        pat, audio = random_pattern(fotf=self.fotf_checked)
        p_good = self.model.eval_audio(audio)
        self.play_pattern(pat, audio, p_good)

    def play_pattern(self, pattern, audio, p_good):
        """
        Play a pattern's audio and show its score
        """

        sd.stop()
        sd.play(audio, 44100)

        self.cur_pattern = pattern
        self.cur_audio = audio

        # Show the score for the new pattern
        self.score_label.configure(text='{:.2f}'.format(p_good))

    def save_good(self):
        if self.cur_audio is None:
            return

        print('Save good pattern')

        assert self.cur_pattern is not None
        assert self.cur_audio is not None

        sd.stop()

        # Save the audio
        save_path = '{}/{:05d}.wav'.format(self.good_path, len(self.good_audio))
        sf.write(save_path, self.cur_audio, 44100)

        self.good_patterns.append(self.cur_pattern)

        # Get audio for pattern (2nd bar)
        audio = self.cur_audio[2*44100:4*44100]
        audio = torch.from_numpy(audio)
        audio = audio.unsqueeze(dim=0)
        self.good_audio.append(audio)
        self.good_label.configure(text=str(len(self.good_audio)))
        self.good_audio_torch = torch.stack(self.good_audio)

        # Pattern saved
        self.cur_pattern = None
        self.cur_audio = None

    def save_bad(self):
        if self.cur_audio is None:
            return

        print('Save bad pattern')

        assert self.cur_pattern is not None
        assert self.cur_audio is not None

        sd.stop()

        # Save the audio
        save_path = '{}/{:05d}.wav'.format(self.bad_path, len(self.bad_audio))
        sf.write(save_path, self.cur_audio, 44100)

        self.good_patterns.append(self.cur_pattern)

        # Get audio for pattern (2nd bar)
        audio = self.cur_audio[2*44100:4*44100]
        audio = torch.from_numpy(audio)
        audio = audio.unsqueeze(dim=0)
        self.bad_audio.append(audio)
        self.bad_label.configure(text=str(len(self.bad_audio)))
        self.bad_audio_torch = torch.stack(self.bad_audio)

        # Pattern saved
        self.cur_pattern = None
        self.cur_audio = None

    def train_loop(self, itr = 0):
        """
        Training loop that runs in the background
        """

        if len(self.good_audio) > 0 and len(self.bad_audio) > 0:
            loss = train_batch(
                self.model,
                self.optimizer,
                self.good_audio_torch,
                self.bad_audio_torch,
                batch_size=self.args.batch_size,
                device=self.device
            )

            if self.running_loss is None:
                self.running_loss = loss
            else:
                self.running_loss = 0.99 * self.running_loss + 0.01 * loss

            print('batch #{}, chunks: {}, running loss: {:.5f}'.format(
                itr + 1,
                (itr + 1) * self.args.batch_size,
                self.running_loss
            ))

            if itr % 50 == 0:
                print('Saving model to "{}"'.format(self.model_path))
                torch.save(self.model.state_dict(), self.model_path)

        self.window.after(
            2,
            lambda: self.train_loop(itr + 1),
        )

if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default='save_data')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--num_conv", type=int, default=64)
    parser.add_argument("--num_mlp", type=int, default=128)
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument("--human_eval", action='store_true')
    args = parser.parse_args()

    deepdrummer = DeepDrummer(args, args.human_eval)

    deepdrummer.event_loop()
