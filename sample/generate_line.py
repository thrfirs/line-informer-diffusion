# This code is based on https://github.com/openai/guided-diffusion
"""
Generate points from a model and save them.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate


def main():
    args = generate_args()

    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')

    n_frames = 5000
    total_num_samples = args.num_samples * args.num_repetitions
    
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    dist_util.setup_dist(args.device)

    params = args.params_for_line
    if params is None:
        raise ValueError('Please specify the parameters for line generation.')
    params = list(map(float, params.split(',')))

    args.num_samples = 1
    args.batch_size = args.num_samples  # Sample a single set of params, so batch size is 1

    print('Loading dataset...')
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=n_frames)

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    collate_args = [{'inp': torch.zeros(n_frames), 'lengths': n_frames, "params": params}] * args.num_samples
    _, model_kwargs = collate(collate_args)
    model_kwargs["y"] = {k: v.to(dist_util.dev()) for k, v in model_kwargs["y"].items()}
    # for k, v in model_kwargs["y"].items():
    #     print(k, type(v))
    
    all_lines = []
    all_params = []
    all_lengths = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i + 1}/{args.num_repetitions}] ###')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (args.batch_size, 5000, 3),  # bs, seq_len, dimension_of_input
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        all_params.append(model_kwargs['y']["params"].cpu().numpy())
        all_lines.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_lines) * args.batch_size} samples")


    all_lines = np.concatenate(all_lines, axis=0)[:total_num_samples]
    all_params = np.concatenate(all_params, axis=0)[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'line': all_lines, 'params': all_params, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    main()
