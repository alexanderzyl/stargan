from generic_solver import GenericSolver
from torchvision.utils import save_image
import torch
import os

from properties import data_on_device
from z_model import ZEncoder, ZDecoder, ZDiscriminator
from z_trainer import ZTrainer


class ZSolver(GenericSolver):
    """Solver for training and testing StarGAN."""

    def __init__(self, loader, config):
        """Initialize configurations."""

        # Data loader.
        super().__init__()
        self.loader = loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard(self.log_dir)

    Encoder = data_on_device()
    Decoder = data_on_device()
    Discriminator = data_on_device()

    def iter_models(self):
        yield self.Encoder
        yield self.Decoder
        yield self.Discriminator

    def build_model(self):
        """Create a generator and a discriminator."""

        def _create_opt(m, lr):
            return torch.optim.Adam(m.parameters(), lr, (self.beta1, self.beta2))

        self.Encoder = ZEncoder(
            name='enc',
            conv_dim=self.g_conv_dim,
            repeat_num=self.g_repeat_num,
            create_optimizer=lambda m: _create_opt(m, self.g_lr))

        self.Decoder = ZDecoder(
            name='dec',
            conv_dim=self.Encoder.out_dim,
            create_optimizer=lambda m: _create_opt(m, self.g_lr))

        self.Discriminator = ZDiscriminator(
            name='D',
            conv_dim=self.Encoder.out_dim,
            c_dim=self.c_dim,
            create_optimizer=lambda m: _create_opt(m, self.d_lr))

        for model in self.iter_models():
            model.print_network()

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        for model in self.iter_models():
            path = os.path.join(self.model_save_dir, '{}-{}.ckpt'.format(resume_iters, model.name))
            model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def get_lrs(self):
        return [m.optimizer.param_groups[0]['lr'] for m in self.iter_models()]

    def update_lrs(self, lrs):
        """Decay learning rates of the generator and discriminator."""
        for lr, model in zip(lrs, self.iter_models()):
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        for m in self.iter_models():
            m.optimizer.zero_grad()

    def train(self):
        tr = ZTrainer(self, self.device)
        tr.train()

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
        data_loader = self.loader

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real, self.generate_image(x_real)]
                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i + 1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

    def generate_image(self, x_real):
        return self.Decoder(self.Encoder(x_real))
