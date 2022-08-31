import os
import numpy as np

from pytorch_lightning.accelerators import accelerator
from models.pose_correction import PoseCorrection
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import *
from models.rendering import *
from models.sdf_utils import *

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
        self.decay_gamma = hparams.decay_gamma
        self.lr_decay = 250
        self.lr_init = hparams.lr
        if hparams.use_sdf:
            self.use_sdf = True
            self.loss = loss_dict['rgbd'](hparams.color_weight,
                                          hparams.depth_weight,
                                          hparams.freespace_weight,
                                          hparams.truncation_weight,
                                          hparams.truncation,
                                          hparams.omni_dir)
        else:
            self.use_sdf = False
            self.loss = loss_dict['color']()

        self.embedding_xyz = Embedding(hparams.N_emb_xyz)
        self.embedding_dir = Embedding(hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        self.nerf_coarse = NeRF(in_channels_xyz=6*hparams.N_emb_xyz+3,
                                in_channels_dir=6*hparams.N_emb_dir+3,
                                omni_dir=hparams.omni_dir)
        self.models = {'coarse': self.nerf_coarse}
        # load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')

        if hparams.N_importance > 0 and not hparams.share_coarse_fine:
            self.nerf_fine = NeRF(in_channels_xyz=6*hparams.N_emb_xyz+3,
                                  in_channels_dir=6*hparams.N_emb_dir+3,
                                  omni_dir=hparams.omni_dir)
            self.models['fine'] = self.nerf_fine
            # load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')
        elif hparams.share_coarse_fine:
            pass


    def forward(self, rays, c2w_array, pose_corr=None):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            c2w_array[:, i:i+self.hparams.chunk],
                            pose_corr,
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            use_sdf=self.hparams.use_sdf,
                            omni_dir=self.hparams.omni_dir,
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
        if self.hparams.max_val_images is not None:
            max_val_images = self.hparams.max_val_images
        else:
            max_val_images = None
        if self.hparams.dataset_name == 'rgbd' and self.hparams.test_train:
            self.train_dataset = dataset(split='test_train', **kwargs)
            self.val_dataset = dataset(split='val', max_val_imgs=max_val_images, **kwargs)
        else:
            self.train_dataset = dataset(split='train', **kwargs)
            self.val_dataset = dataset(split='val', max_val_imgs=max_val_images,**kwargs)
        if not self.hparams.pose_gt:
            self.pose_corr = PoseCorrection(len(self.train_dataset))
            self.models['pose_corr'] = self.pose_corr

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        # scheduler = get_scheduler(self.hparams, self.optimizer)
        # return [self.optimizer], [scheduler]
        self.scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=not self.hparams.no_shuffle,
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
            rays, rgbs, depth, c2ws = batch['rays'], batch['rgbs'], batch['depths'], batch['c2ws']
            results = self(rays, c2ws, self.models['pose_corr'])
            loss = self.loss(results, rgbs)
        else:
            rays, rgbs, depths, c2ws = batch['rays'], batch['rgbs'], batch['depths'], batch['c2ws']
            results = self(rays, c2ws, self.models['pose_corr'] if 'pose_corr' in self.models else None)
            loss, color_fine, depth_fine, fs_coarse, fs_fine, tr_coarse, \
                tr_fine, odf_loss = self.loss(results, rgbs, depths)

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            depth_predicted = results[f'depth_{typ}'].reshape(-1, 1)
            psnr_rgb = psnr(results[f'rgb_{typ}'], rgbs)
            rmse = depth_rmse(depth_predicted, depths)
            rmse_log = depth_rmse_log(depth_predicted, depths)
            abs_rel = depth_abs_rel(depth_predicted, depths)
            sq_rel = depth_sq_rel(depth_predicted, depths)
            delta_1 = depth_delta(depth_predicted, depths, 1)
            delta_2 = depth_delta(depth_predicted, depths, 2)
            delta_3 = depth_delta(depth_predicted, depths, 3)

        self.log('lr', get_learning_rate(self.optimizer), prog_bar=True)
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_rgb, prog_bar=True)
        self.log('train/rmse', rmse, prog_bar=True)
        self.log('train/rmse_log', rmse_log)
        self.log('train/abs_rel', abs_rel)
        self.log('train/sq_rel', sq_rel)
        self.log('train/delta_1', delta_1)
        self.log('train/delta_2', delta_2)
        self.log('train/delta_3', delta_3)
        if self.use_sdf:
            self.log('train/color_loss_fine', color_fine)
            self.log('train/depth_loss_fine', depth_fine)
            self.logger.experiment.add_histogram('train/sdf_fine', results['sigmas_fine'], global_step=self.global_step)
            self.logger.experiment.add_histogram('train/z_vals', results['z_vals_fine'], global_step=self.global_step)
            if self.hparams.omni_dir:
                self.logger.experiment.add_histogram('train/corr_fine', results['corrs_fine'], global_step=self.global_step)
                self.log('train/odf_loss', odf_loss)
            if fs_fine != -1:
                self.log('train/freespace_loss_fine', fs_fine)
                self.log('train/truncation_loss_fine', tr_fine)
            else:
                self.log('train/freespace_loss_coarse', fs_coarse)
                self.log('train/truncation_loss_coarse', tr_coarse)

        self.batch_nb = batch_nb
        self.scheduler.step()       # update learning rate
        return loss

    def validation_step(self, batch, batch_nb):
        if self.use_sdf:
            rays, rgbs, depths, c2ws = batch['rays'], batch['rgbs'], batch['depths'], batch['c2ws']
        else:
            rays, rgbs, depths, c2ws = batch['rays'], batch['rgbs'], batch['depths'], batch['c2ws']
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        if self.use_sdf:
            depths = depths.squeeze() # (H*W, 1)
            results = self(rays, c2ws)
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            depth_predicted = results[f'depth_{typ}'].reshape(-1, 1)
            rmse = depth_rmse(depth_predicted, depths)
            rmse_log = depth_rmse_log(depth_predicted, depths)
            abs_rel = depth_abs_rel(depth_predicted, depths)
            sq_rel = depth_sq_rel(depth_predicted, depths)
            delta_1 = depth_delta(depth_predicted, depths, 1)
            delta_2 = depth_delta(depth_predicted, depths, 2)
            delta_3 = depth_delta(depth_predicted, depths, 3)
            loss, rgb_loss, depth_loss, fs_c, fs_f, tr_c, tr_f, odf_loss = \
                self.loss(results, rgbs, depths)
            log = {
                'val/loss': loss,
                'val/rgb_loss': rgb_loss,
                'val/depth_loss': depth_loss,
                'val/fs_loss': fs_f if fs_f != -1 else fs_c,
                'val/tr_loss': tr_f if tr_f != -1 else tr_c,
                'val/odf_loss': odf_loss,
                'val/rmse': rmse,
                'val/rmse_log': rmse_log,
                'val/abs_rel': abs_rel,
                'val/sq_rel': sq_rel,
                'val/delta_1': delta_1,
                'val/delta_2': delta_2,
                'val/delta_3': delta_3,
            }
            predicted_sdf = results['sigmas_fine']
            index = torch.randint(0, predicted_sdf.shape[0], (1,))
            predicted_sdf = predicted_sdf[index].cpu()
            z_vals = results['z_vals_fine'][index].cpu()
            front_mask, back_mask, sdf_mask = get_gt_sdf_masks(z_vals, depths[index].cpu(),
                                                               self.hparams.truncation)
            gt_sdf = get_gt_sdf(z_vals, depths[index].cpu(), self.hparams.truncation, front_mask, back_mask, sdf_mask)
            fig = plot_sdf_gt_with_predicted(z_vals, gt_sdf, predicted_sdf, depths[index].cpu(), self.hparams.truncation)
            self.logger.experiment.add_image(f'valsdf/sampled_{batch_nb}', fig, global_step=self.global_step)
            if self.hparams.omni_dir:
                self.logger.experiment.add_histogram('val/corr_fine', results['corrs_fine'], global_step=self.current_epoch)
        else:
            depths = depths.squeeze()  # (H*W, 1)
            results = self(rays, c2ws)
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            depth_predicted = results[f'depth_{typ}'].reshape(-1, 1)
            rmse = depth_rmse(depth_predicted, depths)
            rmse_log = depth_rmse_log(depth_predicted, depths)
            abs_rel = depth_abs_rel(depth_predicted, depths)
            sq_rel = depth_sq_rel(depth_predicted, depths)
            delta_1 = depth_delta(depth_predicted, depths, 1)
            delta_2 = depth_delta(depth_predicted, depths, 2)
            delta_3 = depth_delta(depth_predicted, depths, 3)
            log = {
                'val/loss': self.loss(results, rgbs),
                'val/rmse': rmse,
                'val/rmse_log': rmse_log,
                'val/abs_rel': abs_rel,
                'val/sq_rel': sq_rel,
                'val/delta_1': delta_1,
                'val/delta_2': delta_2,
                'val/delta_3': delta_3,
            }

        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        W, H = self.hparams.img_wh
        img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
        depth_gt = visualize_depth(depths.view(H, W)) # (3, H, W)
        stack_rgb = torch.stack([img_gt, img]) # (2, 3, H, W)
        stack_depth = torch.stack([depth_gt, depth]) # (2, 3, H, W)
        stack = torch.cat([stack_rgb, stack_depth], dim=2) # (2, 3, 2H, W)
        self.logger.experiment.add_images(f'val/GT_pred_depth_{batch_nb}',
                                            stack, self.global_step)

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log['val/psnr'] = psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val/loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val/psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr)


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
    logger.experiment.add_text('hparams', str(hparams))

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None,
                      val_check_interval=hparams.val_check_interval,
    )
                    #   strategy=DDPPlugin(find_unused_parameters=False) if hparams.num_gpus>1 else None)

    with torch.autograd.set_detect_anomaly(True):
        trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    print(hparams)
    main(hparams)
