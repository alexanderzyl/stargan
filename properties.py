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