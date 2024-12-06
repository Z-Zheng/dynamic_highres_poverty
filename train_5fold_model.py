# Copyright (c) Stanford Sustainability and AI Lab and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import gc

import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import argparse
import logging
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import torch.distributed as dist

from time import time
import geopandas as gpd
from skimage.io import imread
import albumentations as A
import albumentations.pytorch

import pandas as pd
from torchmetrics.regression import R2Score, MeanSquaredError
from tqdm import tqdm
import numpy as np
import ever as er
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
from models.cnns import EncoderOnly, SiameseEncoderOnly

logger = get_logger(__name__)

ROOTDIR = Path('/scr/zhuo/WB/dataset_v3')

DATASET = {
    'mi_08': {'image_dir': ROOTDIR / 'mi' / '08',
              'awi_geojson': ROOTDIR / 'mi' / 'awi_mi_08.geojson',
              'split_dir': ROOTDIR / 'mi' / 'split'},
    'mi_08_10': {'image_dir': ROOTDIR / 'mi' / '08',
                 'awi_geojson': ROOTDIR / 'mi' / 'awi_10household_mi_08.geojson',
                 'split_dir': ROOTDIR / 'mi' / 'split'},

    'mi_18': {'image_dir': ROOTDIR / 'mi' / '18',
              'awi_geojson': ROOTDIR / 'mi' / 'awi_mi_18.geojson',
              'split_dir': ROOTDIR / 'mi' / 'split'},

    'mi_18_geof': {'image_dir': ROOTDIR / 'mi' / '18',
                   'awi_geojson': ROOTDIR / 'mi' / 'awi_geof_mi_18.geojson',
                   'split_dir': ROOTDIR / 'mi' / 'split'},

    'mi_18_10': {'image_dir': ROOTDIR / 'mi' / '18',
                 'awi_geojson': ROOTDIR / 'mi' / 'awi_10household_mi_18.geojson',
                 'split_dir': ROOTDIR / 'mi' / 'split'},

    'mz_07': {'image_dir': ROOTDIR / 'mz' / '07',
              'awi_geojson': ROOTDIR / 'mz' / 'awi_nonan_inter_mz_07.geojson',
              'split_dir': ROOTDIR / 'mz' / 'split',
              },
    'mz_07_10': {'image_dir': ROOTDIR / 'mz' / '07',
                 'awi_geojson': ROOTDIR / 'mz' / 'awi_10household_nonan_mz_07.geojson',
                 'split_dir': ROOTDIR / 'mz' / 'split',
                 },
    'mz_17': {'image_dir': ROOTDIR / 'mz' / '17',
              'awi_geojson': ROOTDIR / 'mz' / 'awi_nonan_inter_mz_17.geojson',
              'split_dir': ROOTDIR / 'mz' / 'split',
              },

    'mz_17_geof': {'image_dir': ROOTDIR / 'mz' / '17',
                   'awi_geojson': ROOTDIR / 'mz' / 'awi_geof_nonan_inter_mz_17.geojson',
                   'split_dir': ROOTDIR / 'mz' / 'split',
                   },

    'mz_17_10': {'image_dir': ROOTDIR / 'mz' / '17',
                 'awi_geojson': ROOTDIR / 'mz' / 'awi_10household_nonan_mz_17.geojson',
                 'split_dir': ROOTDIR / 'mz' / 'split',
                 },

    'li': {'image_dir': ROOTDIR / 'li' / '04_23_median',
           'awi_geojson': ROOTDIR / 'li' / 'awi_li_04_23.geojson',
           'split_dir': ROOTDIR / 'li' / 'split'},
    'li_10': {'image_dir': ROOTDIR / 'li' / '04_23_median',
              'awi_geojson': ROOTDIR / 'li' / 'awi_10household_li_04_23.geojson',
              'split_dir': ROOTDIR / 'li' / 'split'},

    'li_geof': {'image_dir': ROOTDIR / 'li' / '04_23_median',
                'awi_geojson': ROOTDIR / 'li' / 'awi_geof_li_04_23.geojson',
                'split_dir': ROOTDIR / 'li' / 'split'},

    'li_planetscope': {'image_dir': 'DATA/PlanetScope_Lilongwe_202304/sample',
                       'awi_geojson': ROOTDIR / 'li' / 'awi_li_04_23.geojson',
                       'split_dir': ROOTDIR / 'li' / 'split'},

    'bl': {'image_dir': ROOTDIR / 'bl' / '04_23_median',
           'awi_geojson': ROOTDIR / 'bl' / 'awi_bl_04_23.geojson',
           'split_dir': ROOTDIR / 'bl' / 'split'},
    'bl_10': {'image_dir': ROOTDIR / 'bl' / '04_23_median',
              'awi_geojson': ROOTDIR / 'bl' / 'awi_10household_bl_04_23.geojson',
              'split_dir': ROOTDIR / 'bl' / 'split'},

    'bl_geof': {'image_dir': ROOTDIR / 'bl' / '04_23_median',
                'awi_geojson': ROOTDIR / 'bl' / 'awi_geof_bl_04_23.geojson',
                'split_dir': ROOTDIR / 'bl' / 'split'},
    'bl_planetscope': {'image_dir': 'DATA/PlanetScope_Blantyre_202304/sample',
                       'awi_geojson': ROOTDIR / 'bl' / 'awi_bl_04_23.geojson',
                       'split_dir': ROOTDIR / 'bl' / 'split'},

    'mi_c': {'t1_image_dir': ROOTDIR / 'mi' / '08', 't2_image_dir': ROOTDIR / 'mi' / '18',
             't1_awi_geojson': ROOTDIR / 'mi' / 'awi_10household_mi_08.geojson',
             't2_awi_geojson': ROOTDIR / 'mi' / 'awi_10household_mi_18.geojson',
             'split_dir': ROOTDIR / 'mi' / 'split'
             },
    'mi_c_10': {'t1_image_dir': ROOTDIR / 'mi' / '08', 't2_image_dir': ROOTDIR / 'mi' / '18',
                't1_awi_geojson': ROOTDIR / 'mi' / 'awi_mi_08.geojson', 't2_awi_geojson': ROOTDIR / 'mi' / 'awi_mi_18.geojson',
                'split_dir': ROOTDIR / 'mi' / 'split'
                },

    'mz_c': {'t1_image_dir': ROOTDIR / 'mz' / '07', 't2_image_dir': ROOTDIR / 'mz' / '17',
             't1_awi_geojson': ROOTDIR / 'mz' / 'awi_nonan_inter_mz_07.geojson',
             't2_awi_geojson': ROOTDIR / 'mz' / 'awi_nonan_inter_mz_17.geojson',
             'split_dir': ROOTDIR / 'mz' / 'split'
             },

    'mz_c_10': {'t1_image_dir': ROOTDIR / 'mz' / '07', 't2_image_dir': ROOTDIR / 'mz' / '17',
                't1_awi_geojson': ROOTDIR / 'mz' / 'awi_10household_nonan_mz_07.geojson',
                't2_awi_geojson': ROOTDIR / 'mz' / 'awi_10household_nonan_mz_17.geojson',
                'split_dir': ROOTDIR / 'mz' / 'split'
                },
    'mg': {'image_dir': ROOTDIR / 'mg' / '18',
           'awi_geojson': ROOTDIR / 'mg' / 'awi_nonan_mg_18.geojson',
           'split_dir': ROOTDIR / 'mg' / 'split'},

    'mg_geof': {'image_dir': ROOTDIR / 'mg' / '18',
                'awi_geojson': ROOTDIR / 'mg' / 'awi_geof_nonan_mg_18.geojson',
                'split_dir': ROOTDIR / 'mg' / 'split'},

    'mg_10': {'image_dir': ROOTDIR / 'mg' / '18',
              'awi_geojson': ROOTDIR / 'mg' / 'awi_10household_nonan_mg_18.geojson',
              'split_dir': ROOTDIR / 'mg' / 'split'},

    'bf': {'image_dir': ROOTDIR / 'bf' / '19',
           'awi_geojson': ROOTDIR / 'bf' / 'awi_bf_19.geojson',
           'split_dir': ROOTDIR / 'bf' / 'split'},

    'bf_geof': {'image_dir': ROOTDIR / 'bf' / '19',
                'awi_geojson': ROOTDIR / 'bf' / 'awi_geof_bf_19.geojson',
                'split_dir': ROOTDIR / 'bf' / 'split'},

    'bf_10': {'image_dir': ROOTDIR / 'bf' / '19',
              'awi_geojson': ROOTDIR / 'bf' / 'awi_10household_bf_19.geojson',
              'split_dir': ROOTDIR / 'bf' / 'split'},
}


def stratified_subsample(names, awis, frac=0.1):
    if frac == 1.0:
        return [i for i in range(len(names))]

    assert 0. < frac < 1.
    # create bins
    _, bins = np.histogram(awis, 9)
    labels = np.digitize(awis, bins)

    df = pd.DataFrame({'awi': awis, 'name': names, 'bin': labels})
    sampled_df = df.groupby('bin', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=1))

    return sampled_df.index


def split_train_val(split_dir, val_split, awi_df):
    assert val_split in [0, 1, 2, 3, 4]

    split_dir = Path(split_dir)
    train_names = []
    val_names = []
    for i in range(5):
        if i != val_split:
            train_names += pd.read_csv(split_dir / f'{i}.csv')['name'].tolist()
        else:
            val_names += pd.read_csv(split_dir / f'{i}.csv')['name'].tolist()

    train_awi_df = awi_df[awi_df['name'].isin(train_names)]
    val_awi_df = awi_df[awi_df['name'].isin(val_names)]
    return train_awi_df, val_awi_df


def non_nan_inds(awis):
    inds = []
    for i, awi in enumerate(awis):
        if np.isnan(awi):
            continue
        inds.append(i)
    return inds


class LevelDataset(Dataset):
    fields = ['population', 'pop_density', 'pr', 'tmmn', 'tmmx', 'nightlights', 'gpp_aqua', 'gpp_terra', 'cellid_count',
              'land_cover', 'smod_code', 'building_area', 'building_count', 'ph', 'soc']
    nz_fields = [f'nz_{f}' for f in fields]

    def __init__(self, image_dir, awi_geojson, split_dir, val_split, training=True, train_frac=1.0, load_geofeatures=False):
        df = gpd.read_file(awi_geojson)
        self.load_geofeatures = load_geofeatures

        train_df, val_df = split_train_val(split_dir, val_split, df)
        if training:
            df = train_df
        else:
            df = val_df

        self.names = df['name'].tolist()
        self.awis = df['awi'].tolist()
        if self.load_geofeatures:
            df = df.apply(lambda col: col.fillna(col.mean()) if col.isnull().any() and 'nz_' in col.name else col)
            self.geofeatures = df[self.nz_fields].to_numpy()

        if training:
            valid_inds = non_nan_inds(self.awis)
            self.names = [self.names[i] for i in valid_inds]
            self.awis = [self.awis[i] for i in valid_inds]
            indices = stratified_subsample(self.names, self.awis, frac=train_frac)
            self.names = [self.names[i] for i in indices]
            self.awis = [self.awis[i] for i in indices]
            if self.load_geofeatures:
                self.geofeatures = [self.geofeatures[i] for i in indices]

        self.image_dir = Path(image_dir)

        self.T = A.Compose([
            A.Resize(160, 160),
            A.D4() if training else A.NoOp(),
            A.Normalize(
                mean=[0.0412, 0.0670, 0.0732, 0.2311, 0.2223, 0.1346],
                std=[0.0070, 0.0091, 0.0155, 0.0215, 0.0325, 0.0315],
                max_pixel_value=1.
            ),
            A.pytorch.ToTensorV2()
        ])

    def __getitem__(self, idx):
        name = self.names[idx]
        img = imread(self.image_dir / name)
        awi = self.awis[idx]

        np.putmask(img, np.isnan(img), 0)

        img = self.T(image=img)['image']

        data = {'awi': awi, 'name': name}

        if self.load_geofeatures:
            geof = self.geofeatures[idx]
            geof = torch.from_numpy(geof).reshape(1, -1).to(torch.float32)
            data['geof'] = geof

        return img, data

    def __len__(self):
        return len(self.names)


class ChangeDataset(Dataset):
    def __init__(self, t1_image_dir, t2_image_dir, t1_awi_geojson, t2_awi_geojson,
                 split_dir,
                 val_split,
                 training=True,
                 train_frac=1.0,
                 ):
        df1 = gpd.read_file(t1_awi_geojson)
        df2 = gpd.read_file(t2_awi_geojson)

        train_df1, val_df1 = split_train_val(split_dir, val_split, df1)
        train_df2, val_df2 = split_train_val(split_dir, val_split, df2)
        if training:
            df1 = train_df1
            df2 = train_df2
        else:
            df1 = val_df1
            df2 = val_df2

        merged = df1.merge(df2, how='left', on='name', suffixes=('_t1', '_t2'))
        self.names = merged['name'].tolist()
        self.t1_awis = merged['awi_t1'].tolist()
        self.t2_awis = merged['awi_t2'].tolist()

        if training:
            c_awis = [t2 - t1 for t1, t2 in zip(self.t1_awis, self.t2_awis)]
            valid_inds = non_nan_inds(c_awis)
            self.names = [self.names[i] for i in valid_inds]
            self.t1_awis = [self.t1_awis[i] for i in valid_inds]
            self.t2_awis = [self.t2_awis[i] for i in valid_inds]
            c_awis = [c_awis[i] for i in valid_inds]

            indices = stratified_subsample(self.names, c_awis, frac=train_frac)
            self.names = [self.names[i] for i in indices]
            self.t1_awis = [self.t1_awis[i] for i in indices]
            self.t2_awis = [self.t2_awis[i] for i in indices]

        self.t1_image_dir = Path(t1_image_dir)
        self.t2_image_dir = Path(t2_image_dir)

        self.T = A.Compose([
            A.Resize(160, 160),
            A.D4() if training else A.NoOp(),
            A.Normalize(
                mean=[0.0412, 0.0670, 0.0732, 0.2311, 0.2223, 0.1346],
                std=[0.0070, 0.0091, 0.0155, 0.0215, 0.0325, 0.0315],
                max_pixel_value=1.
            ),
            A.pytorch.ToTensorV2()
        ], additional_targets={'image2': 'image'})

    def __getitem__(self, idx):
        name = self.names[idx]
        img1 = imread(self.t1_image_dir / name)
        img2 = imread(self.t2_image_dir / name)

        np.putmask(img1, np.isnan(img1), 0)
        np.putmask(img2, np.isnan(img2), 0)

        t1_awi = self.t1_awis[idx]
        t2_awi = self.t2_awis[idx]

        data = self.T(image=img1, image2=img2)
        img1 = data['image']
        img2 = data['image2']
        img = torch.cat([img1, img2], dim=0)

        return img, {'t1_awi': t1_awi, 't2_awi': t2_awi, 'awi': t2_awi - t1_awi, 'name': name}

    def __len__(self):
        return len(self.names)


def make_figure(pr, gt, step):
    minv = math.floor(np.min(gt))
    maxv = math.ceil(np.max(gt))
    sns.set(style="ticks")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    sns.scatterplot(x=gt, y=pr, size=0.5, legend=False, alpha=0.4, edgecolor=None)

    x_line = np.linspace(minv, maxv, 100)
    plt.plot(x_line, x_line, color='black', linestyle='--', label='y=x')

    plt.xlim(minv, maxv)
    plt.ylim(minv, maxv)
    plt.xlabel('Wealth Index Ground Truth')
    plt.ylabel('Wealth Index Prediction')

    sns.despine()
    plt.tight_layout()
    wandb.log({"plot": fig}, step=step)


@torch.no_grad()
def evaluate(model, dataloader, device, is_main_process):
    r2score = R2Score().to(device)
    mse = MeanSquaredError().to(device)

    model.eval()

    gts = []
    prs = []
    for img, gt in tqdm(dataloader, disable=is_main_process):
        img = img.to(device)

        gt_awi = gt['awi'].to(device)

        if torch.isnan(gt_awi).item():
            continue

        if 'geof' in gt:
            geof = gt['geof'].to(device)
            pr_awi = model(img, geof)
        else:
            pr_awi = model(img)

        gt_awi = gt_awi.reshape(-1)
        pr_awi = pr_awi.reshape(-1)

        r2score.update(pr_awi, gt_awi)
        mse.update(pr_awi, gt_awi)

        gts.append(gt_awi.cpu().numpy())
        prs.append(pr_awi.cpu().numpy())

    prs = er.dist.all_gather(prs)
    gts = er.dist.all_gather(gts)
    prs = sum(prs, [])
    gts = sum(gts, [])

    gts = np.concatenate(gts, axis=0)
    prs = np.concatenate(prs, axis=0)
    blob = np.stack([gts, prs], axis=0)

    return {
        'eval/r2': r2score.compute().item(),
        'eval/mse': mse.compute().item()
    }, blob


def main(args):
    accelerator_project_config = ProjectConfiguration(project_dir=args.results_dir, logging_dir=args.results_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with='wandb' if args.wandb_project else None,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    device = accelerator.device
    results_dir = Path(args.results_dir)
    if accelerator.is_main_process:
        if args.wandb_project:
            tracker_config = dict(vars(args))
            exp_name = args.exp_name if args.exp_name else f'{args.model}_{args.train_dataset}_frac{int(args.train_frac * 100)}'
            init_kwargs = {'wandb': {'name': exp_name}}
            accelerator.init_trackers(args.wandb_project, config=tracker_config, init_kwargs=init_kwargs)
        results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Experiment directory created at {results_dir}")
    logger.info(accelerator.state, main_process_only=False)

    # model
    if args.model == 'EncoderOnly':
        inc = args.in_channels
        model = EncoderOnly(dict(
            encoder_name=args.backbone,
            in_channels=inc,
            encoder_weights='imagenet',
        ))
    elif args.model == 'TransformerMLP':
        inc = args.in_channels
        from models.transformer import TransformerMLP
        model = TransformerMLP(dict(
            encoder_name=args.backbone,
            in_channels=inc,
            encoder_weights='imagenet',
        ))
    elif args.model == 'MultimodalTransformerMLP':
        from models.transformer import MultimodalTransformerMLP
        model = MultimodalTransformerMLP(dict(
            encoder_name=args.backbone,
            in_channels=args.in_channels,
            g_channels=15,
            encoder_weights='imagenet',
        ))
    elif args.model == 'SiameseEncoderOnly':
        model = SiameseEncoderOnly(dict())
    else:
        raise NotImplementedError

    er.param_util.count_model_parameters(model, logger)
    er.param_util.trainable_parameters(model, logger)

    model = model.to(device)
    model.train()

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.wd)

    if args.train_dataset in ['mi_08', 'mi_18', 'mz_07', 'mz_17', 'bf', 'mg',
                              'mi_08_10', 'mi_18_10',
                              'mz_07_10', 'mz_17_10',
                              'bf_10', 'mg_10',
                              ]:
        train_dataset = LevelDataset(train_frac=args.train_frac,
                                     training=True, val_split=args.val_split, **DATASET[args.train_dataset])
    elif args.train_dataset in ['mi_18_geof', 'mz_17_geof', 'bf_geof', 'mg_geof', ]:
        train_dataset = LevelDataset(
            load_geofeatures=True,
            train_frac=args.train_frac,
            training=True, val_split=args.val_split, **DATASET[args.train_dataset])
    elif args.train_dataset in ['li', 'bl', 'li_geof', 'bl_geof', 'li_planetscope', 'bl_planetscope', 'li_10', 'bl_10']:
        if 'geof' not in args.train_dataset:
            train_dataset = LevelDataset(train_frac=args.train_frac,
                                         training=True, val_split=args.val_split, **DATASET[args.train_dataset])
        else:
            train_dataset = LevelDataset(
                load_geofeatures=True,
                train_frac=args.train_frac,
                training=True, val_split=args.val_split, **DATASET[args.train_dataset])

        if args.train_dataset == 'li_planetscope':
            train_dataset.T = A.Compose([
                A.Resize(128, 128),
                A.D4(),
                A.Normalize(
                    mean=[116.67913816401442, 255.85730741483138, 267.9917078367356, 1316.9370964564562],
                    std=[152.08370010474923, 299.2968177999035, 334.69438880554077, 1493.7774846820237],
                    max_pixel_value=1.
                ),
                A.pytorch.ToTensorV2()
            ])
        elif args.train_dataset == 'bl_planetscope':
            train_dataset.T = A.Compose([
                A.Resize(128, 128),
                A.D4(),
                A.Normalize(
                    mean=[173.06072677991622, 361.3295055401698, 347.6707574663573, 1949.6770461841531],
                    std=[166.28010742337057, 308.3208517103655, 323.56337015717145, 1593.275096446675],
                    max_pixel_value=1.
                ),
                A.pytorch.ToTensorV2()
            ])
        else:
            train_dataset.T = A.Compose([
                A.Resize(512, 512),
                A.D4(),
                A.Normalize(
                    mean=[3987.8290, 4514.7159, 5398.1086, 6225.4530],
                    std=[1803.3756, 1492.8687, 1435.6697, 1800.3449],
                    max_pixel_value=1.
                ),
                A.pytorch.ToTensorV2()
            ])
    elif args.train_dataset in ['mi_c', 'mz_c', 'mi_c_10', 'mz_c_10']:
        train_dataset = ChangeDataset(train_frac=args.train_frac,
                                      training=True, val_split=args.val_split, **DATASET[args.train_dataset])
    else:
        raise NotImplementedError

    if args.val_dataset in ['mi_08', 'mi_18', 'mz_07', 'mz_17', 'bf', 'mg']:
        val_dataset = LevelDataset(training=False, val_split=args.val_split, **DATASET[args.val_dataset])
    elif args.train_dataset in ['mi_18_geof', 'mz_17_geof', 'bf_geof', 'mg_geof', ]:
        val_dataset = LevelDataset(
            load_geofeatures=True,
            training=False, val_split=args.val_split, **DATASET[args.train_dataset])
    elif args.val_dataset in ['li', 'bl', 'li_geof', 'bl_geof', 'li_planetscope', 'bl_planetscope']:
        if 'geof' not in args.train_dataset:
            val_dataset = LevelDataset(training=False, val_split=args.val_split, **DATASET[args.val_dataset])
        else:
            val_dataset = LevelDataset(
                load_geofeatures=True,
                training=False, val_split=args.val_split, **DATASET[args.train_dataset])

        if args.train_dataset == 'li_planetscope':
            val_dataset.T = A.Compose([
                A.Resize(128, 128),
                A.D4(),
                A.Normalize(
                    mean=[116.67913816401442, 255.85730741483138, 267.9917078367356, 1316.9370964564562],
                    std=[152.08370010474923, 299.2968177999035, 334.69438880554077, 1493.7774846820237],
                    max_pixel_value=1.
                ),
                A.pytorch.ToTensorV2()
            ])
        elif args.train_dataset == 'bl_planetscope':
            val_dataset.T = A.Compose([
                A.Resize(128, 128),
                A.D4(),
                A.Normalize(
                    mean=[173.06072677991622, 361.3295055401698, 347.6707574663573, 1949.6770461841531],
                    std=[166.28010742337057, 308.3208517103655, 323.56337015717145, 1593.275096446675],
                    max_pixel_value=1.
                ),
                A.pytorch.ToTensorV2()
            ])
        else:
            val_dataset.T = A.Compose([
                A.Resize(512, 512),
                A.Normalize(
                    mean=[3987.8290, 4514.7159, 5398.1086, 6225.4530],
                    std=[1803.3756, 1492.8687, 1435.6697, 1800.3449],
                    max_pixel_value=1.
                ),
                A.pytorch.ToTensorV2()
            ])
    elif args.val_dataset in ['mi_c', 'mz_c']:
        val_dataset = ChangeDataset(training=False, val_split=args.val_split, **DATASET[args.val_dataset])
    else:
        raise NotImplementedError

    loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    model, opt, loader = accelerator.prepare(model, opt, loader)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False,
        sampler=er.data.DistributedNonOverlapSeqSampler(val_dataset)
    )

    logger.info(f"Dataset contains {len(train_dataset):,} images")

    total_batch_size = args.global_batch_size * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(loader)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {int(args.global_batch_size // accelerator.num_processes)}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.epochs * len(train_dataset) / total_batch_size}")

    best_r2 = 0.
    train_steps = 0
    log_steps = 0
    running_loss = 0
    train_r2 = 0
    start_time = time()
    for epoch in range(args.epochs):
        if train_steps >= args.max_niter:
            break
        logger.info(f"Beginning epoch {epoch}...")
        for img, gt in loader:
            model.train()
            with accelerator.accumulate(model):
                img = img.to(device)
                gt = er.to.to_device(gt, device)

                if 'geof' in gt:
                    geof = gt['geof']
                    msg = model(img, geof.to(img.dtype), gt)
                else:
                    msg = model(img, gt)

                losses = {k: v for k, v in msg.items() if k.endswith('loss')}
                loss = sum([v for _, v in losses.items()])

                opt.zero_grad()
                accelerator.backward(loss)
                opt.step()

                running_loss += loss.item()
                train_r2 += msg['train/r2'].item()
                if accelerator.sync_gradients:
                    log_steps += 1
                    train_steps += 1

                loss_info = {k: v.detach().item() for k, v in losses.items()}
                logs = {
                    "train/loss": loss.detach().item(),
                    "train/r2": msg['train/r2'].item(),
                    "lr": args.lr
                }
                logs.update(loss_info)

                if train_steps % args.log_every == 0:
                    accelerator.log(logs, step=train_steps)
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / accelerator.num_processes

                    avg_train_r2 = torch.tensor(train_r2 / log_steps, device=device)
                    dist.all_reduce(avg_train_r2, op=dist.ReduceOp.SUM)
                    avg_train_r2 = avg_train_r2.item() / accelerator.num_processes

                    logger.info(
                        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train R2: {avg_train_r2:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    train_r2 = 0
                    start_time = time()

        gc.collect()
        if (epoch + 1) % args.val_epoch == 0:
            results, blob = evaluate(model, val_loader, accelerator.device, accelerator.is_main_process)
            if results['eval/r2'] > best_r2:
                best_r2 = results['eval/r2']
                np.save(Path(args.results_dir) / f'{args.val_split}_gt_pr.npy', blob)
                if accelerator.is_main_process and args.save_best_model and args.train_frac == 1.0:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{results_dir}/model_best_{args.val_split}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            er.dist.synchronize()

            if er.dist.is_main_process() and args.wandb_project is not None:
                make_figure(blob[1], blob[0], train_steps)
            results.update(step=train_steps)
            logger.info(results)
            accelerator.log(results, step=train_steps)
            gc.collect()

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info("Done!")
        logger.info(f'best R^2: {best_r2}')

    accelerator.end_training()
    return best_r2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)

    parser.add_argument("--model", type=str, default='EncoderOnly')
    parser.add_argument("--backbone", type=str, default='resnet18')
    parser.add_argument("--in_channels", type=int, default=6)

    parser.add_argument("--train_dataset", type=str, default=None)
    parser.add_argument("--val_dataset", type=str, default=None)
    parser.add_argument("--val_split", type=int, default=0)
    parser.add_argument("--val_epoch", type=int, default=2)
    parser.add_argument("--train_frac", type=float, default=1.0)

    parser.add_argument("--global_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_niter", type=int, default=10000000)

    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--save_best_model", action='store_true')
    parser.add_argument("--only_full", action='store_true')

    args = parser.parse_args()
    global_batch_size = args.global_batch_size
    data = dict(model=[], site=[], frac=[], r2=[], fold=[])
    results_dir = Path(args.results_dir)

    if (results_dir / 'results.csv').exists():
        df = pd.read_csv(results_dir / 'results.csv')
    else:
        df = None

    if args.only_full:
        fracs = [1.0]
    else:
        fracs = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

    for k in [0, 1, 2, 3, 4]:
        for frac in fracs:
            args.val_split = k
            args.train_frac = frac
            if frac == 0.01:
                args.global_batch_size = 16
                if args.train_dataset.startswith('bl'):
                    args.global_batch_size = 15
            else:
                args.global_batch_size = global_batch_size

            if df is not None:
                if ((df['fold'] == k) & (int(frac * 100) == df['frac'])).any():
                    print(f'skip. fold: {k}, frac: {frac}')
                    continue

            r2 = main(args)
            gc.collect()

            data['fold'].append(k)
            data['model'].append(args.model_name)
            data['site'].append(args.val_dataset)
            data['frac'].append(int(frac * 100))
            data['r2'].append(r2)

            if df is not None:
                pd.concat([df, pd.DataFrame(data=data)]).to_csv(results_dir / 'results.csv', index=False)
            else:
                pd.DataFrame(data=data).to_csv(results_dir / 'results.csv', index=False)
