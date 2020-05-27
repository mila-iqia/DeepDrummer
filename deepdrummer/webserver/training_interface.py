"""
Interface code for WEB UI and training setup.
"""

import sys
import time
from pathlib import Path
from uuid import uuid4
from multiprocessing import Process, Queue, Pipe
import numpy as np
import soundfile as sf
from deepdrummer.experiment import Experiment
from . import APP

# Map of user indices to experiment objects
experiments = {}

def train_process(pipe, exp_kwargs):
    """
    Training process. Communicates through a bidirectional pipe
    """

    # Create the experiment object
    experiment = Experiment(**exp_kwargs)

    # Keep track of the current phase
    phase = 1

    # Last time a message was received
    last_msg = time.time()

    while True:
        # If we haven't received any messages for 30 minutes, stop the process
        if time.time() - last_msg > 30 * 60:
            print('Training process timed out')
            return

        print('Receiving message')

        # Read the message
        req = pipe.recv()
        req_type = req[0]
        args = req[1:]

        last_msg = time.time()

        print('got request: ', req)
        sys.stdout.flush()

        if req_type == 'ping':
            continue

        if req_type == 'add_rating':
            print('got rating')
            sys.stdout.flush()

            if phase is 1:
                experiment.add_rating_phase1(args[0], args[1])
            else:
                experiment.add_rating_phase2(args[0], args[1])
            continue

        if req_type == 'gen_clip':
            if phase is 1:
                clip = experiment.gen_clip_phase1()
                pipe.send(clip)
            else:
                clip = experiment.gen_clip_phase2()
                pipe.send(clip)
            continue

        if req_type == 'start_phase2':
            print('received start of phase 2')
            experiment.start_phase2()
            phase = 2
            continue

        if req_type == 'save_data':
            survey_data = args[0]
            save_path = args[1]
            data = experiment.save_data(survey_data, save_path)
            pipe.send(data)
            continue

        if req_type == 'close':
            break

        assert False, "unknown request type " + req[0]


class WebExperiment():
    """
    Handle web-specific details of experiment management
    """

    def __init__(self, **kwargs):

        self.pipe, pipe_child = Pipe()

        self.proc = Process(target=train_process, kwargs={
            'pipe': pipe_child,
            'exp_kwargs': kwargs
        })

        self.proc.start()

        # Generate an audio clip
        self.pipe.send(['gen_clip'])
        self.last_clip = self.pipe.recv()

        self.last_clip_id = None

        self.phase = 1

    def __del__(self):
        """
        Kill the process when deleting the experiment object
        """

        print('**** DELETING EXPERIMENT OBJECT ****')

        self.pipe.send(['close'])
        self.pipe.close()
        self.proc.join()

    @property
    def is_running(self):
        """
        Check if the training process is still running
        """

        try:
            self.pipe.send(['ping'])
        except IOError as e:
            return False

        return True

    def gen_clip(self, out_file_path):
        """
        Create audio and fill in audio specified WAV file.

        :param out_file_path: Path to the WAV which should be filled.
        :returns: The identifier of the clip.
        :rtype: str
        """

        # If in phase 2, generate the next clip now
        if self.phase is 2:
            self.pipe.send(['gen_clip'])
            self.last_clip = self.pipe.recv()

        out_file_path = Path(out_file_path)
        APP.logger.debug("Placing audio data into file %s", out_file_path)
        sf.write(out_file_path, self.last_clip, 44100)

        self.last_clip_id = uuid4()

        return self.last_clip_id

    def add_rating(self, clip_uuid, rating):
        # In the web (async) case we want to prevent double submissions
        if str(clip_uuid) != str(self.last_clip_id):
            raise KeyError('mismatched clip uuid')

        if self.phase is 1:
            # Generate the next audio clip now to hide latency
            self.pipe.send(['gen_clip'])
            next_clip = self.pipe.recv()

            # Send the rating to the training process
            self.pipe.send(['add_rating', rating, self.last_clip])

            self.last_clip = next_clip
        else:
            # Send the rating
            self.pipe.send(['add_rating', rating, self.last_clip])

    def start_phase2(self):
        """
        Signal the start of phase 2
        """

        self.pipe.send(['start_phase2'])
        self.phase = 2

    def save_pause_survey(self, survey_data):
        self.pause_survey_data = survey_data

    def save_data(self, survey_data, out_path):
        """
        Save the experiment data
        """

        # Merge the data from both surveys
        survey_data = {**self.pause_survey_data, **survey_data}

        self.pipe.send(['save_data', survey_data, out_path])
        data = self.pipe.recv()
        return data
