import os
import copy
import math
import numpy as np

class Pattern:
    """
    Represent a grid pattern with M instruments/samples over N-steps
    """

    def literal(*args):
        """
        Constructor to parse an array literal notation (manual initialization)
        """

        instr_names = []
        steps = []

        num_steps = 0

        for arg in args:
            if isinstance(arg, str):
                instr_names.append(arg)
            else:
                assert len(steps) == 0 or len(arg) == len(steps[0])
                steps.append(arg)
                num_steps = len(arg)

        return Pattern(num_steps, instr_names, steps)

    def __init__(self, num_steps, instr_names, init=None):
        """
        Empty pattern constructor
        """

        self.num_steps = num_steps
        self.num_instrs = len(instr_names)
        self.instr_names = instr_names

        # Internal numpy array to store the pattern
        if init:
            self.array = np.array(init, dtype=np.bool)
            assert self.array.shape[0] == self.num_instrs
            assert self.array.shape[1] == self.num_steps
        else:
            self.array = np.zeros(shape=(self.num_instrs, self.num_steps), dtype=np.bool)

    @property
    def count(self):
        return np.count_nonzero(self.array)

    def set(self, instr:int, step:int, val=True):
        self.array[instr, step] = val

    def get(self, instr, step):
        return self.array[instr, step]

    def randomize(self, p=0.5):
        """
        Produce a random pattern
        """

        self.array = np.random.rand(self.num_instrs, self.num_steps) < p

    def copy(self):
        return copy.deepcopy(self)

    def mutate(self, p=0.05):
        """
        Mutate this pattern
        """

        mut_toggle = np.random.rand(self.num_instrs, self.num_steps) < p
        not_pat = np.logical_not(self.array)
        self.array = np.where(mut_toggle, not_pat, self.array)

    def mutate_samples(self, p=0.15):
        """
        Mutate the samples used by this pattern
        """

        for idx in range(self.num_instrs):
            if np.random.uniform(0, 1) < p:
                self.instr_names[idx] = SampleManager.random(1)[0]

    def dist(pat_a, pat_b):
        """
        Compute the edit distance between two patterns
        """

        diff = np.bitwise_xor(pat_a.array, pat_b.array)
        return np.count_nonzero(diff)

    def __str__(self):
        """
        Produce a string representation of the pattern
        """

        instrLen = max(map(len, self.instr_names))

        out = ''

        for instr_idx in range(self.num_instrs):
            name = self.instr_names[instr_idx]
            if len(name) < instrLen:
                name += ' ' * (instrLen - len(name))
            out += name + ' '

            out += '['
            for step_idx in range(self.num_steps):
                out += 'X' if self.array[instr_idx, step_idx] else ' '
                if step_idx < self.num_steps - 1:
                    out += ','
            out += ']'

            if instr_idx < self.num_instrs - 1:
                out += '\n'

        return out

    def render(self, num_repeats=4, bpm=120, sr=44100, pad_len=1, mix_vol=0.5):
        """
        Render the pattern using audio samples
        Produces a numpy array
        """

        # Load the samples
        samples = [SampleManager.load(name) for name in self.instr_names]

        steps_per_beat = 4
        beat_len = (60 / bpm)
        note_len = beat_len / steps_per_beat

        # Allocate a buffer to generate the audio
        pat_len = num_repeats * self.num_steps * note_len + pad_len
        num_samples = int(pat_len * sr)
        audio = np.zeros(shape=num_samples, dtype=np.float32)

        for step_idx in range(num_repeats * self.num_steps):
            start_idx = int(step_idx * note_len * sr)
            step_idx = step_idx % self.num_steps

            for instr_idx in range(self.num_instrs):
                if self.get(instr_idx, step_idx):
                    mix_sample(audio, samples[instr_idx] * mix_vol, start_idx)

        return audio

class SampleManager:
    """
    Find, load and cache samples
    """

    # Cache of loaded samples, indexed by name
    cache = {}

    @classmethod
    def root_dir(self):
        """
        Get the root directory where samples are located
        """

        mod_path, _ = os.path.split(os.path.realpath(__file__))
        samples_path = os.path.realpath(os.path.join(mod_path, '..', 'samples'))
        return samples_path

    @classmethod
    def get_list(self, prefix=None):
        """
        Get the list of names of all available samples
        """

        # If the list has not yet been compiled
        if not hasattr(self, 'name_list'):
            root_path = self.root_dir()

            names = []

            for file_root, dirs, files in os.walk(root_path, topdown=False):
                for name in files:
                    name, ext = name.split('.')
                    if ext != 'wav':
                        continue

                    relpath = os.path.relpath(file_root, root_path)
                    name = os.path.join(relpath, name)
                    names.append(name)

            setattr(self, 'name_list', names)

        names = getattr(self, 'name_list')

        if prefix:
            names = list(filter(lambda s: s.startswith(prefix), names))

        return names

    @classmethod
    def get_path(self, name):
        """
        Get the absolute path of an audio sample file located in /samples
        """
        samples_path = self.root_dir()
        file_path = os.path.join(samples_path, name + '.wav')

        return file_path

    @classmethod
    def load(self, name):
        """
        Load a sample into a numpy array
        """

        if name in self.cache:
            return self.cache[name]

        import soundfile as sf
        path = self.get_path(name)
        data, sr = sf.read(path)
        assert sr == 44100

        # Mix down to mono if necessary
        if len(data.shape) == 2 and data.shape[1] == 2:
            data = 0.5 * (data[:, 0] + data[:, 1])

        data = data.astype(np.float32)

        self.cache[name] = data

        return data

    @classmethod
    def load_all(self, prefix=None):
        """
        Load all the samples into one NumPy array
        Note that the total size of the array depends on the longest sample
        """

        sample_list = self.get_list(prefix)

        samples = []

        for name in sample_list:
            data = self.load(name)
            samples.append((name, data, data.shape[0]))

        # Get the length of the longest sample
        samples = sorted(samples, key=lambda e: e[-1])
        num_samples = len(samples)
        max_len = samples[-1][-1]

        # Array to store the samples
        arr = np.zeros(shape=(num_samples, max_len), dtype=np.float32)

        for idx, (_, sample, smp_len) in enumerate(samples):
            arr[idx, :smp_len] = sample

        return arr

    @classmethod
    def random(self, num_samples, prefix=None):
        import random
        samples = self.get_list(prefix)
        names = random.sample(samples, num_samples)
        return names

def mix_sample(audio, sample, start_idx):
    """
    Mix an audio sample into an audio buffer in-place
    """

    smp_len = sample.shape[0]

    end_idx = start_idx + smp_len
    if end_idx > audio.shape[-1]:
        end_idx = audio.shape[-1]

    smp_len = end_idx - start_idx

    audio[start_idx:end_idx] = audio[start_idx:end_idx] + sample[:smp_len]

##############################################################################

def new_pattern(fotf):
    samples = SampleManager.random(4)

    if fotf:
        samples[0] = SampleManager.random(1, 'drumhits/Kick')[0]

    pat = Pattern(16, samples)
    return pat

def random_pattern(p=0.5, fotf=False):
    pat = new_pattern(fotf)
    pat.randomize(p)

    # Force 4 on the floor pattern for first sample
    if fotf:
        for i in range(pat.num_steps):
            pat.set(0, i, i % 4 == 0)

    audio = pat.render()
    return pat, audio

def mutate_pattern(pat, fotf=False):
    """
    Mutate a pattern, while optionally preserving 4-on-the-floor structure
    """

    new_pattern = pat.copy()
    new_pattern.mutate()
    new_pattern.mutate_samples()

    if fotf:
        # Make sure the top row pattern remains 4oft
        for i in range(new_pattern.num_steps):
            new_pattern.set(0, i, i % 4 == 0)

        # Make sure the first sample remains a kick
        if new_pattern.instr_names[0] != pat.instr_names[0]:
            new_pattern.instr_names[0] = SampleManager.random(1, 'drumhits/Kick')[0]

    return new_pattern

##############################################################################

def sample_prior(model, fotf, target_score=1.0, num_itrs=50):
    """
    Sample audio using a structured pattern generation prior
    """

    best_pattern = None
    best_audio = None
    best_p_good = None
    best_dist = math.inf

    for i in range(num_itrs):
        print(i)
        pat, audio = gen_prior(fotf)
        p_good = model.eval_audio(audio)
        dist = abs(target_score - p_good)

        if dist <= best_dist:
            best_pattern = pat
            best_audio = audio
            best_p_good = p_good
            best_dist = dist

    return best_pattern, best_audio, best_p_good

def sample_metro(model, fotf, min_itrs=100, max_itrs=500, min_p=0, verbose=False):
    """
    Sample from a distribution using the Metropolis-Hastings algorithm
    """

    cur_pattern, cur_audio = random_pattern(fotf)
    cur_p = model.eval_audio(cur_audio)

    for i in range(max_itrs):
        new_pattern = mutate_pattern(cur_pattern, fotf=fotf)
        new_audio = new_pattern.render()
        new_p = model.eval_audio(new_audio)

        a = new_p / cur_p

        if a >= 1 or np.random.uniform(0, 1) < a:
            if verbose:
                print(i, new_p)
            cur_pattern = new_pattern
            cur_audio = new_audio
            cur_p = new_p

        if i+1 >= min_itrs and cur_p > min_p:
            break

    return cur_pattern, cur_audio, cur_p

def hillclimb(model, fotf, target_score=1.0, target_dist=0.05, min_itrs=100, max_itrs=500, verbose=False):
    """
    Try maximizing fitness through greedy hill-climbing
    """

    best_pattern, best_audio = random_pattern(fotf)
    best_p_good = None
    best_dist = math.inf

    for i in range(max_itrs):
        new_pattern = mutate_pattern(best_pattern, fotf=fotf)
        audio = new_pattern.render()
        p_good = model.eval_audio(audio)
        dist = abs(target_score - p_good)

        if dist <= best_dist:
            if verbose:
                print(i, p_good)
            best_pattern = new_pattern
            best_audio = audio
            best_p_good = p_good
            best_dist = dist

            if i >= min_itrs and best_dist <= target_dist:
                break

    return best_pattern, best_audio, best_p_good
