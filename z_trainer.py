import torch
import time
import datetime
import os
import torch.nn.functional as F

from torchvision.utils import save_image

from properties import data_on_device


def classification_loss(logit, target):
    """Compute binary or softmax cross entropy loss."""
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


class ZTrainer:

    def __init__(self, solver, device):
        self.solver = solver
        self.device = device

    # data to be stored on gpu
    x_fixed = data_on_device()
    x_real = data_on_device()
    c_org = data_on_device()
    c_trg = data_on_device()
    alpha = data_on_device()
    label_org = data_on_device()  # Labels for computing classification loss.
    label_trg = data_on_device()  # Labels for computing classification loss.

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)  # noqa
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        data_loader = self.solver.loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        self.x_fixed, c_org = next(data_iter)
        c_fixed_list = self.solver.create_labels(c_org, self.solver.c_dim, self.solver.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.solver.g_lr
        d_lr = self.solver.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.solver.resume_iters:
            start_iters = self.solver.resume_iters
            self.solver.restore_model(self.solver.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.solver.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                self.x_real, self.label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                self.x_real, self.label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(self.label_org.size(0))
            self.label_trg = self.label_org[rand_idx]

            self.c_org = self.label_org.clone()  # Original domain labels.
            self.c_trg = self.label_trg.clone() # Target domain labels.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            loss = self.train_discriminator()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.solver.n_critic == 0:
                g_loss = self.train_generator()
                loss = {**loss, **g_loss}

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.solver.log_step == 0:
                self.print_train_info(i, loss, start_time)

            # Translate fixed images for debugging.
            if (i + 1) % self.solver.sample_step == 0:
                with torch.no_grad():
                    self.save_fixed_images(c_fixed_list, i)

            # Save model checkpoints.
            if (i + 1) % self.solver.model_save_step == 0:
                self.save_checkpoint(i)

            # Decay learning rates.
            if (i + 1) % self.solver.lr_update_step == 0 and (i + 1) > (
                    self.solver.num_iters - self.solver.num_iters_decay):
                g_lr -= (self.solver.g_lr / float(self.solver.num_iters_decay))
                d_lr -= (self.solver.d_lr / float(self.solver.num_iters_decay))
                self.solver.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def save_checkpoint(self, i):
        g_path = os.path.join(self.solver.model_save_dir, '{}-G.ckpt'.format(i + 1))
        d_path = os.path.join(self.solver.model_save_dir, '{}-D.ckpt'.format(i + 1))
        torch.save(self.solver.G.state_dict(), g_path)
        torch.save(self.solver.D.state_dict(), d_path)
        print('Saved model checkpoints into {}...'.format(self.solver.model_save_dir))

    def save_fixed_images(self, c_fixed_list, i):
        x_fake_list = [self.x_fixed]
        for c_fixed in c_fixed_list:
            x_fake_list.append(self.solver.G(self.x_fixed, c_fixed))
        x_concat = torch.cat(x_fake_list, dim=3)
        sample_path = os.path.join(self.solver.sample_dir, '{}-images.jpg'.format(i + 1))
        save_image(self.solver.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
        print('Saved real and fake images into {}...'.format(sample_path))

    def print_train_info(self, i, loss, start_time):
        et = time.time() - start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.solver.num_iters)
        for tag, value in loss.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)
        if self.solver.use_tensorboard:
            for tag, value in loss.items():
                self.solver.logger.scalar_summary(tag, value, i + 1)

    def train_generator(self):
        # Original-to-target domain.
        x_fake = self.solver.G(self.x_real, self.c_trg)
        out_src, out_cls = self.solver.D(x_fake)
        g_loss_fake = - torch.mean(out_src)
        g_loss_cls = classification_loss(out_cls, self.label_trg)
        # Target-to-original domain.
        x_reconst = self.solver.G(x_fake, self.c_org)
        g_loss_rec = torch.mean(torch.abs(self.x_real - x_reconst))
        # Backward and optimize.
        g_loss = g_loss_fake + self.solver.lambda_rec * g_loss_rec + self.solver.lambda_cls * g_loss_cls
        self.solver.reset_grad()
        g_loss.backward()
        self.solver.G.optimizer.step()
        # Logging.
        g_loss = {'G/loss_fake': g_loss_fake.item(), 'G/loss_rec': g_loss_rec.item(),
                  'G/loss_cls': g_loss_cls.item()}
        return g_loss

    def train_discriminator(self):
        # Compute loss with real images.
        out_src, out_cls = self.solver.D(self.x_real)
        d_loss_real = - torch.mean(out_src)
        d_loss_cls = classification_loss(out_cls, self.label_org)
        # Compute loss with fake images.
        x_fake = self.solver.G(self.x_real, self.c_trg)
        out_src, out_cls = self.solver.D(x_fake.detach())
        d_loss_fake = torch.mean(out_src)
        # Compute loss for gradient penalty.
        self.alpha = torch.rand(self.x_real.size(0), 1, 1, 1)
        x_hat = (self.alpha * self.x_real.data + (1 - self.alpha) * x_fake.data).requires_grad_(True)
        out_src, _ = self.solver.D(x_hat)
        d_loss_gp = self.gradient_penalty(out_src, x_hat)
        # Backward and optimize.
        d_loss = d_loss_real + d_loss_fake + self.solver.lambda_cls * d_loss_cls + self.solver.lambda_gp * d_loss_gp
        self.solver.reset_grad()
        d_loss.backward()
        self.solver.D.optimizer.step()
        # Logging.
        loss = {'D/loss_real': d_loss_real.item(), 'D/loss_fake': d_loss_fake.item(),
                'D/loss_cls': d_loss_cls.item(), 'D/loss_gp': d_loss_gp.item()}
        return loss
