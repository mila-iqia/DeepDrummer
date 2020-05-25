import os
import argparse
import soundfile as sf
from .patterns import *
from .models import *

class MultiModelMean:
    def __init__(self, models):
        self.models = models
        assert len(self.models) > 0

    def eval_audio(self, audio):
        p_goods = [model.eval_audio(audio) for model in self.models]
        p_good = sum(p_goods) / len(self.models)
        return p_good

class MultiModelVote:
    def __init__(self, models):
        self.models = models
        assert len(self.models) > 0

    def eval_audio(self, audio):
        p_goods = [1 if model.eval_audio(audio) > 0.5 else 0 for model in self.models]
        p_good = sum(p_goods) / len(self.models)
        return p_good

def load_models(data_dir_path, device='cuda'):
    models = []

    for root, dirs, files in os.walk(data_dir_path, topdown=False):
        for name in dirs:
            model_path = os.path.join(root, name, 'model_080.pt')
            print(model_path)

            if not os.path.exists(model_path):
                continue

            model = MFCCCritic(
                n_conv=64,
                n_mlp=128,
                p_dropout=0.16744183752066633
            )
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            models.append(model)

    return models

if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./web_data')
    parser.add_argument("--out_path", default='.')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--max_itrs", type=int, default=5000)
    parser.add_argument("--num_gen", type=int, default=20)
    parser.add_argument("--num_topk", type=int, default=5)
    args = parser.parse_args()

    models = load_models(args.data_dir, device=args.device)
    multi_model = MultiModelMean(models)
    print('loaded {} models'.format(len(models)))

    clips = []

    # Try to generate good clips
    for i in range(args.num_gen):
        print('Generating good clip {}/{}'.format(i+1, args.num_gen))
        pattern, audio, p_good = hillclimb(multi_model, fotf=False, target_score=1.0, target_dist=0.05, min_itrs=1000, max_itrs=args.max_itrs)
        clips.append((audio, p_good))

    # Try to generate bad clips
    for i in range(args.num_gen):
        print('Generating bad clip {}/{}'.format(i+1, args.num_gen))
        pattern, audio, p_good = hillclimb(multi_model, fotf=False, target_score=0.0, target_dist=0.01, min_itrs=1000, max_itrs=args.max_itrs)
        clips.append((audio, p_good))

    clips.sort(key=lambda t: t[-1])
    best_clips = clips[-args.num_topk:]
    worst_clips = clips[:args.num_topk]

    for idx, (audio, p_good) in enumerate(best_clips):
        save_path = '{}/good_{:03d}_p{:.2f}.wav'.format(args.out_path, idx, p_good)
        sf.write(save_path, audio, 44100)

    for idx, (audio, p_good) in enumerate(worst_clips):
        save_path = '{}/bad_{:03d}_p{:.2f}.wav'.format(args.out_path, idx, p_good)
        sf.write(save_path, audio, 44100)
