import os
import json
import time
import numpy as np
import soundfile as sf
import torch
from .patterns import sample_metro, hillclimb
from .models import *
from .training import *

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

class Experiment:
    """
    API to manage a single DeepDrummer experiment
    This API tries to be independent of user interface concerns
    """

    def __init__(
        self,
        user_email,
        user_name,
        phase1_count=200,
        save_interval=10,
        trials_per_model=8,
        verbose=True,

        # Hyperparams from first pass
        # 4535
        train_batches=40,
        batch_size=16,
        lr=5.73932300111413e-05,
        weight_decay=5.8959660787385e-06,
        dropout=0.16744183752066633,

        num_conv=64,
        num_mlp=128,
        fotf=False,
    ):
        assert phase1_count % save_interval == 0

        # Training parameters
        self.train_batches = train_batches
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.num_conv = num_conv
        self.num_mlp = num_mlp
        self.fotf = fotf

        # NOTE: we intentionally perform no e-mail and name validation here
        # because the Experiment class needs to be usable in an
        # offline/dummy context
        self.user_email = user_email
        self.user_name = user_name
        self.start_time = time.time()
        self.end_time = None
        self.phase1_count = phase1_count
        self.save_interval = save_interval
        self.trials_per_model = trials_per_model
        self.verbose = verbose

        # Current phase of the experiment
        self.phase = 1

        self.phase1_ratings = []
        self.phase2_ratings = []

        self.phase1_clips = []
        self.phase2_clips = []

        # Audio clips labeled as good and bad in phase 1
        self.good_audio = []
        self.bad_audio = []

        self.good_audio_torch = None
        self.bad_audio_torch = None

        self.train_loss = []

        # Create a new, untrained, freshly initialized model
        self.model = self.new_model()

        # Optimizer for the critic model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Models saved during training
        self.saved_models = [(0, self.new_model(self.model))]

    def new_model(self, copy_model=None):
        """
        Create a new untrained model
        """

        model = MFCCCritic(
            n_conv=self.num_conv,
            n_mlp=self.num_mlp,
            p_dropout=self.dropout
        )

        model.cuda()

        # If we should copy the parameters from another model
        if copy_model is not None:
            model.load_state_dict(copy_model.state_dict())

        return model

    def gen_clip_phase1(self):
        """
        Generate audio for the first phase of the experiment
        """

        pattern, audio, p = sample_metro(self.model, fotf=self.fotf, min_itrs=100)

        return audio

    def _train_incremental(self):
        """
        Do incremental retraining with additional ratings
        """

        if len(self.good_audio) == 0 or len(self.bad_audio) == 0:
            return

        for itr in range(self.train_batches):
            loss = train_batch(
                self.model,
                self.optimizer,
                self.good_audio_torch,
                self.bad_audio_torch,
                batch_size=self.batch_size,
                device=self.model.device
            )

            if self.verbose:
                print('batch #{}, loss: {:.5f}'.format(
                    itr + 1,
                    loss
                ))

            self.train_loss.append(loss)

    def add_rating_phase1(self, rating, audio):
        """
        Add a new rating for one clip
        """

        assert rating in ['good', 'bad']

        def stack(audio_clips):
            audio = np.stack(audio_clips)

            print(audio.shape)

            audio = torch.from_numpy(audio)
            audio = audio[:, 2*44100:4*44100]
            audio = audio.unsqueeze(dim=1)

            print(audio.shape)

            return audio

        if rating == 'good':
            self.good_audio.append(audio)
            self.good_audio_torch = stack(self.good_audio)
        else:
            self.bad_audio.append(audio)
            self.bad_audio_torch = stack(self.bad_audio)

        self.phase1_clips.append(audio)
        self.phase1_ratings.append([rating, time.time()])

        # Train the model with the new rating
        self._train_incremental()

        # Save the current model every save interval
        if len(self.phase1_ratings) % self.save_interval == 0:
            self.saved_models.append((len(self.phase1_ratings), self.new_model(self.model)))

    def start_phase2(self):
        """
        This happens just before the pause and begins
        the second phase of the experiment.
        """

        assert len(self.phase1_ratings) == self.phase1_count
        assert len(self.saved_models) > 0

        # Generate the clips to be rated in phase 2
        for model_count, model in self.saved_models:
            for i in range(self.trials_per_model):
                pattern, audio, p = sample_metro(model, fotf=self.fotf, min_itrs=100, min_p=0.95)
                self.phase2_clips.append((model_count, audio))
                print('generating clip from model {} {}/{}'.format(model_count, i+1, self.trials_per_model))

        # Randomly shuffle the generated clips so they are not shown in order
        np.random.shuffle(self.phase2_clips)

        self.phase = 2

    def gen_clip_phase2(self):
        """
        Generate audio for the second phase of the experiment
        """

        assert len(self.phase2_ratings) < len(self.phase2_clips)

        clip_idx = len(self.phase2_ratings)
        model_count, audio = self.phase2_clips[clip_idx]

        return audio

    def add_rating_phase2(self, rating, rated_audio):
        """
        Add a new rating for one clip
        """

        assert rating in ['good', 'bad']
        assert len(self.phase2_ratings) < len(self.phase2_clips)

        clip_idx = len(self.phase2_ratings)
        model_count, audio = self.phase2_clips[clip_idx]

        # Make sure the rating corresponds to the right clip
        assert np.array_equal(audio, rated_audio)

        self.phase2_ratings.append([model_count, rating, time.time()])

    def save_data(self, survey_data, out_path):
        """
        Package and save the data from this experiment
        We want to save all audio and ratings
        """

        print('saving experiment data to "{}"'.format(out_path))

        # Check that all clips were rated
        assert len(self.phase1_ratings) == self.phase1_count
        assert len(self.phase2_ratings) == len(self.phase2_clips)

        # Store the time when the experiment ended
        self.end_time = time.time()

        # Create the directory where the data will be saved
        # This way, we can be sure that the new directory is empty
        os.makedirs(out_path)

        # Sort phase2_ratings by training sample count
        self.phase2_ratings.sort(key=lambda v: v[0])

        # Save the training parameters
        params = {}
        for key in dir(self):
            if 'time' in key:
                continue

            value = getattr(self, key)
            if isinstance(value, (int, float, bool)):
                params[key] = value

        data = {
            'params': params,
            'user_email': self.user_email,
            'user_name': self.user_name,
            'survey_data': survey_data,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'phase1_ratings': self.phase1_ratings,
            'phase2_ratings': self.phase2_ratings,
            'train_loss': self.train_loss
        }

        json_path = os.path.join(out_path, "data.json")
        with open(json_path, "w") as outfile:
            json.dump(data, outfile)

        # Save the phase 1 clips
        for clip_idx, clip in enumerate(self.phase1_clips):
            clip_path = '{}/phase1_{:03d}.wav'.format(out_path, clip_idx)
            sf.write(clip_path, clip, 44100)

        # Save the phase 2 clips
        for clip_idx, (model_count, clip) in enumerate(self.phase2_clips):
            clip_path = '{}/phase2_{:03d}.wav'.format(out_path, clip_idx)
            sf.write(clip_path, clip, 44100)

        # Save the trained models
        for model_count, model in self.saved_models:
            if hasattr(model, 'state_dict'):
                model_path = os.path.join(out_path, "model_{:03d}.pt".format(model_count))
                torch.save(model.state_dict(), model_path)

        return data
