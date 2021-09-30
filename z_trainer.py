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
        lrs = self.solver.get_lrs()
        delta_lrs = [lr / float(self.solver.num_iters_decay) for lr in self.solver.get_lrs()]

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
            self.c_trg = self.label_trg.clone()  # Target domain labels.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # loss = self.train_discriminator()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            # if (i + 1) % self.solver.n_critic == 0:
            #     g_loss = self.train_generator()
            #     loss = {**loss, **g_loss}

            loss = self.train_generator()

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
                lrs = [lr - d for lr, d in zip(lrs, delta_lrs)]
                self.solver.update_lrs(lrs)
                print('Decayed learning rates:')
                for lr, model in zip(lrs, self.solver.iter_models()):
                    print("Model: {}, lr = {}".format(model.name, lr))

    def save_checkpoint(self, i):
        for model in self.solver.iter_models():
            path = os.path.join(self.solver.model_save_dir, '{}-{}.ckpt'.format(i + 1, model.name))
            torch.save(model.state_dict(), path)
            # save for the forward prop
            gpt_path = os.path.join(self.solver.model_save_dir, '{}-{}.pt'.format(i + 1, model.name))
            sm = torch.jit.script(model)
            sm.save(gpt_path)

        print('Saved model checkpoints into {}...'.format(self.solver.model_save_dir))

    def save_fixed_images(self, c_fixed_list, i):
        x_fake_list = [self.x_fixed]
        for c_fixed in c_fixed_list:
            x_fake_list.append(self.solver.Decoder(self.solver.Encoder(self.x_fixed, c_fixed)))
        x_concat = torch.cat(x_fake_list, dim=3)
        sample_path = os.path.join(self.solver.sample_dir, '{}-images.jpg'.format(i + 1))
        save_image(self.solver.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
        print('Saved real and fake images into {}...'.format(sample_path))

    def save_fixed_images_old(self, c_fixed_list, i):
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
        # Original-to-feature domain.
        z_vector = self.solver.Encoder(self.x_real, self.c_trg)
        # Feature-to-reconstructed domain.
        x_reconst = self.solver.Decoder(z_vector)
        g_loss_rec = self.solver.lambda_rec * torch.mean(torch.abs(self.x_real - x_reconst))
        # Backward and optimize.
        self.solver.reset_grad()
        g_loss_rec.backward()
        self.solver.Encoder.optimizer.step()
        self.solver.Decoder.optimizer.step()
        # Logging.
        g_loss = {'G/loss_rec': g_loss_rec.item()}
        return g_loss

    def train_discriminator(self):
        # Compute loss with real images.
        z_vector = self.solver.Encoder(self.x_real)
        out_cls = self.solver.D(z_vector)
        d_loss_cls = classification_loss(out_cls, self.label_org)
        # Compute loss for gradient penalty.
        self.alpha = torch.rand(self.x_real.size(0), 1, 1, 1)
        # Backward and optimize.
        d_loss = self.solver.lambda_cls * d_loss_cls
        self.solver.reset_grad()
        d_loss.backward()
        self.solver.D.optimizer.step()
        # Logging.
        loss = {'D/loss_cls': d_loss_cls.item()}
        return loss
