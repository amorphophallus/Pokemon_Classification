import torch
import pandas as pd
from PIL import Image
import os
from torchvision.transforms import transforms
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, img_dir, data_frame):
        """
        :param img_dir: 图片目录
        :param data_frame: 图片名字和标签的列表，需要包含列 'Name', 'Type1', 'Type2'
        :other dataset_expansion: 每个图片经过数据增广之后生成多少张图片
        :other transform: 用于数据增广的操作
        :other index_to_label: 用于标签的编号到字符串的转换
        :other label_to_index: 用于标签的字符串到编号的转换
        """
        self.img_dir = img_dir
        self.data_frame = data_frame
        self.dataset_expansion = 10  # 通过 image augmentation 生成多少倍数量的图像
        self.transform = transforms.Compose([  # image augmentation
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33)),  # 随机裁剪，顺便调整大小
            transforms.RandomHorizontalFlip(p=0.5),  # 以 0.5 的概率水平翻转
            transforms.RandomRotation(90),  # 随机旋转 (-90, +90) 度
            transforms.ColorJitter(brightness=0.3, contrast=0, saturation=0, hue=0),  # 随机改变图像的亮度、对比度、饱和度和色调
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.index_to_label = {
            0: 'Bug',
            1: 'Dark',
            2: 'Dragon',
            3: 'Electric',
            4: 'Fairy',
            5: 'Fighting',
            6: 'Fire',
            7: 'Flying',
            8: 'Ghost',
            9: 'Grass',
            10: 'Ground',
            11: 'Ice',
            12: 'Normal',
            13: 'Poison',
            14: 'Psychic',
            15: 'Rock',
            16: 'Steel',
            17: 'Water'
        }
        self.label_to_index = {
            'Bug': 0,
            'Dark': 1,
            'Dragon': 2,
            'Electric': 3,
            'Fairy': 4,
            'Fighting': 5,
            'Fire': 6,
            'Flying': 7,
            'Ghost': 8,
            'Grass': 9,
            'Ground': 10,
            'Ice': 11,
            'Normal': 12,
            'Poison': 13,
            'Psychic': 14,
            'Rock': 15,
            'Steel': 16,
            'Water': 17
        }

    def __len__(self):
        return len(self.data_frame) * self.dataset_expansion

    def __getitem__(self, idx):
        real_idx = idx // self.dataset_expansion

        # get image after augmentation
        pokemon_name = self.data_frame.iloc[real_idx].at['Name']
        if os.path.exists('pokemon/images/' + pokemon_name + '.png'):
            file_name = 'pokemon/images/' + pokemon_name + '.png'
            img = Image.open(file_name).convert('RGB')  # PNG 是 RGBA 4 通道的，需要先转成 3 通道再进行 transform
        elif os.path.exists('pokemon/images/' + pokemon_name + '.jpg'):
            file_name = 'pokemon/images/' + pokemon_name + '.jpg'
            img = Image.open(file_name)
        else:
            print("no such image file!")
            exit(1)
        aug_img = self.transform(img)

        # get multi-label
        label = torch.zeros((18, ))
        label[self.label_to_index[self.data_frame.iloc[real_idx].at['Type1']]] = 1
        if pd.notnull(self.data_frame.iloc[real_idx].at['Type2']):
            label[self.label_to_index[self.data_frame.iloc[real_idx].at['Type2']]] = 1

        return aug_img, label

    def get_num_classes(self):
        return len(self.label_to_index)


if __name__ == '__main__':
    origin_df = pd.read_csv('pokemon/pokemon.csv')  # load original data
    train_df = pd.read_csv('pokemon/train.csv').merge(origin_df, on="Name")
    test_df = pd.read_csv('pokemon/test.csv').merge(origin_df, on="Name")
    dataset_train = CustomDataset('pokemon/images', train_df)
    dataset_test = CustomDataset('pokemon/images', test_df)
    print(dataset_train[0])
    print(dataset_train[40])
