"""
Run with:
python3 -m deepdrummer.prepare_materials
"""

import os
import shutil
import json
import argparse
import soundfile as sf
from deepgroove.deepdrummer.training import load_audio_files
from deepgroove.deepdrummer.multimodel import MultiModelMean, MultiModelVote, load_models
from deepgroove.deepdrummer.patterns import hillclimb

def anonymize(data_dir):
    """
    Remove names and emails from the JSON data
    """

    for root, dirs, files in os.walk(dst_data_dir, topdown=False):
        for dir_name in sorted(dirs):
            dir_path = os.path.join(root, dir_name)
            json_path = os.path.join(root, dir_name, 'data.json')
            print(json_path)

            with open(json_path) as f:
                data = json.load(f)

            data.pop('user_email')
            data.pop('user_name')
            data.pop('survey_data')

            with open(json_path, 'w') as f:
                json.dump(data, f)

def find_best_worst(data_dir, out_dir, num_gen=30, num_topk=5, max_itrs=5000):
    """
    Find the best and worst clips based on an ensemble of all models
    """

    all_clips = load_audio_files(data_dir, single_bar=False)
    all_clips = [clip.squeeze().numpy() for clip in all_clips]

    print('loaded {} clips'.format(len(all_clips)))

    models = load_models(data_dir)
    multi_model = MultiModelMean(models)
    print('Loaded {} models'.format(len(models)))

    clips_w_scores = []

    # Produce scores for all clips
    for idx, clip in enumerate(all_clips):
        print('evaluating clip {} / {}'.format(idx+1, len(all_clips)))
        score = multi_model.eval_audio(clip[2*44100:4*44100])
        clips_w_scores.append((clip, score))

    clips_w_scores.sort(key=lambda t: t[-1])
    
    best_5 = reversed(clips_w_scores[-5:])
    worst_5 = clips_w_scores[:5]

    for idx, (clip, score) in enumerate(best_5):
        print(score)
        dst_path = os.path.join(out_dir, 'best_{}_p{:.2f}.wav'.format(idx+1, score))
        sf.write(dst_path, clip, 44100)
    print()

    for idx, (clip, score) in enumerate(worst_5):
        print(score)
        dst_path = os.path.join(out_dir, 'worst_{}_p{:.2f}.wav'.format(idx+1, score))
        sf.write(dst_path, clip, 44100)
    print()

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--samples_dir", default='samples')
parser.add_argument("--data_dir", default='data')
parser.add_argument("--dst_dir", default='dd_materials')
args = parser.parse_args()

assert os.path.exists(args.data_dir)

if os.path.exists(args.dst_dir):
    shutil.rmtree(args.dst_dir)
os.mkdir(args.dst_dir)

dst_data_dir = os.path.join(args.dst_dir, 'data')
best_worst_dir = os.path.join(args.dst_dir, 'best_worst_loops')
os.mkdir(best_worst_dir)

# Copy the samples
shutil.copytree(args.samples_dir, os.path.join(args.dst_dir, 'samples'))

# Copy the experiment data
shutil.copytree(args.data_dir, dst_data_dir)

# Anoymize the data
anonymize(dst_data_dir)

# Find the best and worst clips
find_best_worst(args.data_dir, best_worst_dir)
