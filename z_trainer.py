import torch
import time
import datetime
import os

from torchvision.utils import save_image


class data_on_device(property):

    def __init__(self):
        super().__init__(self.__get_x, self.__set_x)
        self._x = None

    def __get_x(self, _owner):
        return self._x

    def __set_x(self, _owner, v):
        if _owner.device is not None:
            self._x = v.to(_owner.device)
        else:
            self._x = v


class ZTrainer:

    def __init__(self, solver, device):
        self.solver = solver
        self.device = device

    # data to be stored on gpu
    x_fixed = data_on_device()

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
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = label_org.clone()
            c_trg = label_trg.clone()

            x_real = x_real.to(self.solver.device)  # Input images.
            c_org = c_org.to(self.solver.device)  # Original domain labels.
            c_trg = c_trg.to(self.solver.device)  # Target domain labels.
            label_org = label_org.to(self.solver.device)  # Labels for computing classification loss.
            label_trg = label_trg.to(self.solver.device)  # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.solver.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.solver.classification_loss(out_cls, label_org)

            # Compute loss with fake images.
            x_fake = self.solver.G(x_real, c_trg)
            out_src, out_cls = self.solver.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.solver.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.solver.D(x_hat)
            d_loss_gp = self.solver.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.solver.lambda_cls * d_loss_cls + self.solver.lambda_gp * d_loss_gp
            self.solver.reset_grad()
            d_loss.backward()
            self.solver.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.solver.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.solver.G(x_real, c_trg)
                out_src, out_cls = self.solver.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.solver.classification_loss(out_cls, label_trg)

                # Target-to-original domain.
                x_reconst = self.solver.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.solver.lambda_rec * g_loss_rec + self.solver.lambda_cls * g_loss_cls
                self.solver.reset_grad()
                g_loss.backward()
                self.solver.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.solver.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.solver.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.solver.use_tensorboard:
                    for tag, value in loss.items():
                        self.solver.logger.scalar_summary(tag, value, i + 1)

            # Translate fixed images for debugging.
            if (i + 1) % self.solver.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [self.x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.solver.G(self.x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.solver.sample_dir, '{}-images.jpg'.format(i + 1))
                    save_image(self.solver.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i + 1) % self.solver.model_save_step == 0:
                G_path = os.path.join(self.solver.model_save_dir, '{}-G.ckpt'.format(i + 1))
                D_path = os.path.join(self.solver.model_save_dir, '{}-D.ckpt'.format(i + 1))
                torch.save(self.solver.G.state_dict(), G_path)
                torch.save(self.solver.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.solver.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.solver.lr_update_step == 0 and (i + 1) > (
                    self.solver.num_iters - self.solver.num_iters_decay):
                g_lr -= (self.solver.g_lr / float(self.solver.num_iters_decay))
                d_lr -= (self.solver.d_lr / float(self.solver.num_iters_decay))
                self.solver.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
