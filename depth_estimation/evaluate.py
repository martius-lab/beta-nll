import argparse
import os
import math
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

import model_io
from dataloader import DepthDataLoader
from models import UnetAdaptiveBins, UnetGaussian
from utils import RunningAverageDict


IMAGES_TO_SAVE = {
    'bookstore__rgb_00084',
    'bookstore__rgb_00088',
    'bedroom__rgb_00170',
    'bedroom__rgb_00946',
    'bedroom__rgb_00934',
    'bedroom__rgb_00077',
    'classroom__rgb_00299',
    'classroom__rgb_00309',
    'computer_lab__rgb_00333',
    'dining_room__rgb_01399',
    'dining_room__rgb_01442',
    'living_room__rgb_01339',
    'living_room__rgb_01303',
    'office__rgb_00635',
    'office__rgb_00634',
    'study__rgb_00470',
    'study_room__rgb_00272',
}


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


_LOG_2PI = math.log(2 * math.pi)


def compute_nll(gt, pred, std):
    ll = -0.5 * ((gt - pred) ** 2 / (std**2) + 2 * np.log(std) + _LOG_2PI)
    return dict(nll=-np.mean(ll, axis=-1))

# def denormalize(x, device='cpu'):
#     mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
#     std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
#     return x * std + mean
#
def predict_tta(model, image, args):
    res = model(image)
    pred = res[-1]
    #     pred = utils.depth_norm(pred)
    #     pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
    #     pred = np.clip(pred.cpu().numpy(), 10, 1000)/100.
    pred = np.clip(pred.cpu().numpy(), args.min_depth, args.max_depth)

    image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(device)

    res_lr = model(image)
    pred_lr = res_lr[-1]
    #     pred_lr = utils.depth_norm(pred_lr)
    #     pred_lr = nn.functional.interpolate(pred_lr, depth.shape[-2:], mode='bilinear', align_corners=True)
    #     pred_lr = np.clip(pred_lr.cpu().numpy()[...,::-1], 10, 1000)/100.
    pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], args.min_depth, args.max_depth)

    final = 0.5 * (pred + pred_lr)
    final = nn.functional.interpolate(torch.Tensor(final), image.shape[-2:], mode='bilinear', align_corners=True)

    if args.model == "UnetGaussian":
        var = res[0].cpu()
        var_lr = torch.from_numpy(res_lr[0].cpu().numpy()[..., ::-1].copy())
        var_final = 0.5 * (var + var_lr)
        var_final = nn.functional.interpolate(var_final, image.shape[-2:], mode='bilinear', align_corners=True)
        return torch.Tensor(final), var_final.sqrt()
    else:
        return torch.Tensor(final), None


def eval(model, test_loader, args, gpus=None, log_dir=None):
    if gpus is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = gpus[0]

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        print(f'Logging to {log_dir}')

    metrics = RunningAverageDict()
    # crop_size = (471 - 45, 601 - 41)
    # bins = utils.get_bins(100)
    total_invalid = 0
    with torch.no_grad():
        model.eval()

        sequential = test_loader
        for batch in tqdm(sequential):

            image = batch['image'].to(device)
            gt = batch['depth'].to(device)
            final, std = predict_tta(model, image, args)
            final = final.squeeze().cpu().numpy()

            # final[final < args.min_depth] = args.min_depth
            # final[final > args.max_depth] = args.max_depth
            final[np.isinf(final)] = args.max_depth
            final[np.isnan(final)] = args.min_depth

            if log_dir is not None:
                if args.dataset == 'nyu':
                    impath = f"{batch['image_path'][0].replace('/', '__').replace('.jpg', '')}"
                    factor = 1000
                else:
                    dpath = batch['image_path'][0].split('/')
                    impath = dpath[1] + "_" + dpath[-1]
                    impath = impath.split('.')[0]
                    factor = 256

                # rgb_path = os.path.join(rgb_dir, f"{impath}.png")
                # tf.ToPILImage()(denormalize(image.squeeze().unsqueeze(0).cpu()).squeeze()).save(rgb_path)

                pred_path = os.path.join(log_dir, f"{impath}.png")
                pred = (final * factor).astype('uint16')
                if impath in IMAGES_TO_SAVE:
                    Image.fromarray(pred).save(pred_path)

                if std is not None:
                    std_path = os.path.join(log_dir, f"{impath}_std.npy")
                    # std_path = os.path.join(log_dir, f"{impath}_std.png")
                    std_np = std.squeeze().numpy()
                    if impath in IMAGES_TO_SAVE:
                        np.save(std_path, std_np)
                    #std = np.clip(std, args.min_depth, args.max_depth)
                    #std = (std * factor).astype('uint16')
                    # Image.fromarray(std).save(std_path)

            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    # print("Invalid ground truth")
                    total_invalid += 1
                    continue

            gt = gt.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt > args.min_depth, gt < args.max_depth)

            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
            #             gt = gt[valid_mask]
            #             final = final[valid_mask]

            metrics.update(compute_errors(gt[valid_mask], final[valid_mask]))
            if std is not None:
                std = std.squeeze().cpu().numpy()
                metrics.update(compute_nll(gt[valid_mask], final[valid_mask], std[valid_mask]))

    print(f"Total invalid: {total_invalid}")
    metrics = {k: v for k, v in metrics.get_value().items()}
    metrics_rounded = {k: round(v, 3) for k, v in metrics.items()}
    print(f"Metrics: {metrics_rounded}")

    if log_dir is not None:
        head = ','.join(f'{key}' for key in metrics.keys())
        values = ','.join(f'{value}' for value in metrics.values())
        with open(os.path.join(log_dir, 'metrics.csv'), 'w') as f:
            f.write(head + '\n' + values + '\n')


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Model evaluator', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--n-bins', '--n_bins', default=256, type=int,
                        help='number of bins/buckets to divide depth range into')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument('--save-dir', '--save_dir', default=None, type=str, help='Store predictions in folder')
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")

    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")

    parser.add_argument("--data_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset gt")

    parser.add_argument('--filenames_file',
                        default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
                        type=str, help='path to the filenames text file')

    parser.add_argument('--input_height', type=int, help='input height', default=416)
    parser.add_argument('--input_width', type=int, help='input width', default=544)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)

    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')

    parser.add_argument('--data_path_eval',
                        default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval', default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the groundtruth data for online evaluation')
    parser.add_argument('--filenames_file_eval',
                        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file for online evaluation')
    parser.add_argument('--checkpoint_path', '--checkpoint-path', type=str, required=True,
                        help="checkpoint file to use for prediction")

    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('--do_kb_crop', help='Use kitti benchmark cropping', action='store_true')

    parser.add_argument("--model", default="UnetAdaptiveBins", help="Model type")

    if sys.argv.__len__() >= 2 and os.path.isfile(sys.argv[1]):
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix] + sys.argv[2:])
    else:
        args = parser.parse_args()

    # args = parser.parse_args()
    args.gpu = int(args.gpu) if args.gpu is not None else 0
    args.distributed = False
    device = torch.device('cuda:{}'.format(args.gpu))
    test = DepthDataLoader(args, 'online_eval').data

    if args.model == "UnetAdaptiveBins":
        model = UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                       norm='linear')
    elif args.model == "UnetGaussian":
        model = UnetGaussian.build(min_val=args.min_depth, max_val=args.max_depth)
    elif args.model == "Unet1D":
        model = UnetGaussian.build(min_val=args.min_depth, max_val=args.max_depth, gaussian=False)
    else:
        raise ValueError(f"Unknown model type {args.model}")

    model = model.to(device)
    model = model_io.load_checkpoint(args.checkpoint_path, model)[0]
    model = model.eval()

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        checkpoint_name = pathlib.Path(args.checkpoint_path).stem
        log_dir = os.path.join(args.save_dir, checkpoint_name)

    eval(model, test, args, gpus=[device], log_dir=log_dir)
