import torch
from abc import ABCMeta, abstractmethod


class GenericSolver(object, metaclass=ABCMeta):

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    @abstractmethod
    def train(self):
        """Train StarGAN within a single dataset."""
        pass

    @abstractmethod
    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        pass

    def create_labels(self, c_org, c_dim=5, selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def build_tensorboard(self, log_dir):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(log_dir)

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)
