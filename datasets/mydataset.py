import paddle
import numpy as np
import cv2


class MyDataset(paddle.io.Dataset):
    def __init__(self, mode, transform, img_size=(1024, 1024)):
        super(MyDataset, self).__init__()
        assert mode in ['train', 'val', 'test'], "mode is one of ['train', 'val', 'test']"
        self.mode = mode
        self.transform = transform
        self.img_size = img_size
        self.data = []
        self._load_data(mode)

    def _load_data(self, mode):
        with open(mode + '.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line.split(' ')) == 2:
                    self.data.append([
                        line.split(' ')[0],
                        line.split(' ')[1]
                    ])
                elif len(line.split(' ')) == 1:
                    self.data.append([
                        line
                    ])

    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, self.img_size)
        return img

    def __getitem__(self, idx):
        if len(self.data[idx]) == 2:
            img = self._load_image(self.data[idx][0])
            t_label = self._load_image(self.data[idx][1])
            label = cv2.cvtColor(t_label, cv2.COLOR_BGR2GRAY)
            # label = np.expand_dims(label, axis=0)
            label = np.clip(label, 0, 1)
            if self.mode == 'train' or 'val':
                img, label = self.transform(img), label
            img = img.astype('float32')
            label = label.astype('int64')
            return paddle.to_tensor(img), paddle.to_tensor(label)
        if len(self.data[idx]) == 1:
            img = self._load_image(self.data[idx][0])
            img = self.transform(img)
            img = img.astype('float32')
            return img

    def __len__(self):
        return len(self.data)
    