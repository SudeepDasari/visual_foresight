class IMTopic:
    def __init__(self, name, height=640, width=480, top=0, bot=0, right=0, left=0, dtype="bgr8", flip=False):
        self._name = name
        self._width = width
        self._top = top
        self._bot = bot
        self._right = right
        self._left = left
        self._dtype = dtype
        self._height = height
        self._flip = flip

    def process_image(self, img):
        assert self._bot + self._top < img.shape[0], "Overcrop! bot + top crop >= image height!"
        assert self._right + self._left < img.shape[1], "Overcrop! right + left crop >= image width!"

        bot, left = self._bot, self._left
        if self._bot <= 0:
            bot = img.shape[0] + 10
        if self._left <= 0:
            left = img.shape[1] + 10

        img = img[self._top:bot, self._right:left]
        if self.flip:
            img = img[::-1, ::-1]

        return img

    @property
    def name(self):
        return self._name

    @property
    def width(self):
        return self._width

    @property
    def top(self):
        return self._top

    @property
    def bot(self):
        return self._bot

    @property
    def right(self):
        return self._right

    @property
    def left(self):
        return self._left

    @property
    def dtype(self):
        return self._dtype

    @property
    def height(self):
        return self._height

    @property
    def flip(self):
        return self._flip