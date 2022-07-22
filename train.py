import os
import numpy as np

from pytorch_lightning.accelerators import accelerator
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import *
from models.rendering import *

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        if hparams.use_sdf:
            self.use_sdf = True
            self.loss = loss_dict['rgbd'](hparams.color_weight,
                                          hparams.depth_weight,
                                          hparams.freespace_weight,
                                          hparams.truncation_weight,
                                          hparams.truncation)
        else:
            self.use_sdf = False
            self.loss = loss_dict['color'](coef=1)

        self.embedding_xyz = Embedding(hparams.N_emb_xyz)
        self.embedding_dir = Embedding(hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        self.nerf_coarse = NeRF(in_channels_xyz=6*hparams.N_emb_xyz+3,
                                in_channels_dir=6*hparams.N_emb_dir+3)
        self.models = {'coarse': self.nerf_coarse}
        load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')

        if hparams.N_importance > 0:
            self.nerf_fine = NeRF(in_channels_xyz=6*hparams.N_emb_xyz+3,
                                  in_channels_dir=6*hparams.N_emb_dir+3)
            self.models['fine'] = self.nerf_fine
            load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            use_sdf=self.hparams.use_sdf
                            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        if self.hparams.dataset_name == 'rgbd' and self.hparams.test_train:
            self.train_dataset = dataset(split='test_train', **kwargs)
            self.val_dataset = dataset(split='val', max_val_imgs=1, **kwargs)
        else:
            self.train_dataset = dataset(split='train', **kwargs)
            self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        # scheduler = get_scheduler(self.hparams, self.optimizer)
        # return [self.optimizer], [scheduler]
        return [self.optimizer], ExponentialLR(self.optimizer, gamma=0.1)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        if not self.use_sdf:
            rays, rgbs = batch['rays'], batch['rgbs']
            results = self(rays)
            loss = self.loss(results, rgbs)
        else:
            rays, rgbs, depths = batch['rays'], batch['rgbs'], batch['depths']
            results = self(rays)
            loss, color_fine, depth_fine, fs_coarse, tr_coarse, \
                fs_fine, tr_fine = self.loss(results, rgbs, depths)

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_rgb = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        self.log('train/psnr_rgb', psnr_rgb, prog_bar=True)
        if self.use_sdf:
            self.log('train/color_loss_fine', color_fine, prog_bar=True)
            self.log('train/depth_loss_fine', depth_fine)
            if fs_fine != -1:
                self.log('train/freespace_loss_fine', fs_fine)
                self.log('train/truncation_loss_fine', tr_fine)
            else:
                self.log('train/freespace_loss_coarse', fs_coarse)
                self.log('train/truncation_loss_coarse', tr_coarse)

        return loss

    def validation_step(self, batch, batch_nb):
        if self.use_sdf:
            rays, rgbs, depths = batch['rays'], batch['rgbs'], batch['depths']
        else:
            rays, rgbs = batch['rays'], batch['rgbs']
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        if self.use_sdf:
            depths = depths.squeeze() # (H*W, 1)
            results = self(rays)
            loss, rgb_loss, depth_loss, fs_c, tr_c, fs_f, tr_f = \
                self.loss(results, rgbs, depths)
            log = {
                'val/loss': loss,
                'val/rgb_loss': rgb_loss,
                'val/depth_loss': depth_loss,
                'val/freespace_loss': fs_f if fs_f != -1 else fs_c,
                'val/truncation_loss': tr_f if tr_f != -1 else tr_c,
            }
        else:
            results = self(rays)
            log = {'val/loss': self.loss(results, rgbs)}

        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        W, H = self.hparams.img_wh
        img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
        stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
        self.logger.experiment.add_images(f'val/GT_pred_depth_{batch_nb}',
                                            stack, self.global_step)

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log['val/psnr'] = psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val/loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val/psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)


def main(hparams):
    system = NeRFSystem(hparams)
    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              monitor='val/psnr',
                              mode='max',
                              save_top_k=5)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='auto',
                      devices=hparams.num_gpus,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None,
                      strategy=DDPPlugin(find_unused_parameters=False) if hparams.num_gpus>1 else None)

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    print(hparams)
    main(hparams)
