from abc import ABCMeta, abstractmethod


class GenericSolver(object, metaclass=ABCMeta):
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