_device = 1


class data_on_device(property):

    def __init__(self, device):
        super().__init__(self.__get_x, self.__set_x)
        self.device = device
        self._x = None

    def __get_x(self, _owner):
        return self._x

    def __set_x(self, _owner, v):
        if self.device is not None:
            self._x = v.to(self.device)
        else:
            self._x = v


class ZTrainer:

    def __init__(self):
        pass

    x_fixed = data_on_device(_device)
    y_fixed = data_on_device(_device)


test = ZTrainer()
test.x_fixed = 4
print(test.x_fixed)

test.y_fixed = 3
print(test.y_fixed)
