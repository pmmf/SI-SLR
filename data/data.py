import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage import io
from sklearn.model_selection import (ShuffleSplit, StratifiedShuffleSplit,
                                     GroupShuffleSplit)
from torch.utils.data import Dataset
from torchvision import transforms

import random
from random import randint
import copy

sys.path.insert(0, './utils/')
from utils import one_hot_1D, one_hot_2D, inverse_transform
sys.path.insert(0, './data/')
from transforms import (numpyToTensor, Normalize_1_1, Normalize_0_1,
                        randomHorizontalFlip, randomAffineTransform,
                        randomColourTransform, ImgAugTransform)

DATA_TRANSF = transforms.Compose([Normalize_0_1(),
                                  Normalize_1_1(),
                                  numpyToTensor()])


def getSplitter(dataset, n_splits=5, mode='groups', test_size=.1):
    '''get data splitters'''
    X_to_split = np.zeros((len(dataset), 1))
    if mode == 'groups':
        rs = GroupShuffleSplit(n_splits=n_splits, test_size=test_size,
                               random_state=42)
        g = [dataset[i][2] for i in range(len(dataset))]
        return rs.split(X_to_split, groups=g)
    elif mode == 'stratify':
        rs = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                    random_state=42)

        y = [dataset[i][1] for i in range(len(dataset))]
        return rs.split(splitter.split(X_to_split, y))
    elif type(mode) == list:
        tr_indexes = [i for i in range(len(dataset)) if dataset[i][2] in mode]
        test_indexes = [i for i in range(len(dataset))
                        if dataset[i][2] not in mode]
        return [(tr_indexes, test_indexes)]
    else:
        rs = ShuffleSplit(n_splits=n_splits, test_size=test_size,
                          random_state=42)
        return rs.split(X_to_split)


def split_data(dataset, splitter, batch_sze, dataAug, mode='groups'):
    '''Split train and test'''
    X_to_split = np.zeros((len(dataset), 1))

    tr_indexes, test_indexes = splitter
    g_train = list(np.unique([dataset[i][2] for i in tr_indexes]))
    g_test = list(np.unique([dataset[i][2] for i in test_indexes]))

    # train, test and validation loaders
    data_train_tmp = KinectLeap(model=dataset.model, isTrain=True,
                                person_ids=g_train, dataAug=False)

    if type(mode) == list:
        mode = 'groups'
    splitter = getSplitter(data_train_tmp, n_splits=1, mode=mode, test_size=.1)

    for split, (tr_indexes, valid_indexes) in enumerate(splitter):
        g_train = list(np.unique([data_train_tmp[i][2] for i in tr_indexes]))
        g_valid = list(np.unique([data_train_tmp[i][2] for i in valid_indexes]))
        break

    print("g_train: {}\n g_valid: {}\n g_test: {}".format(g_train, g_valid,
                                                          g_test))
    data_train = dataset.__class__(model=dataset.model, isTrain=True,
                                   person_ids=g_train, dataAug=dataAug)
    data_valid = dataset.__class__(model=dataset.model, isTrain=False,
                                   person_ids=g_valid, dataAug=False)
    data_test = dataset.__class__(model=dataset.model, isTrain=False,
                                  person_ids=g_test, dataAug=False)

    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=batch_sze,
                                               shuffle=True, num_workers=4,
                                               sampler=None)

    valid_loader = torch.utils.data.DataLoader(data_valid,
                                               batch_size=batch_sze,
                                               shuffle=False, num_workers=4,
                                               sampler=None)

    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=batch_sze,
                                              shuffle=False, num_workers=4,
                                              sampler=None)

    return train_loader, valid_loader, test_loader


class KinectLeap(Dataset):
    def __init__(self,
                 data_fn='/data/DB/kinect_leap_dataset_signer_independent/',
                 n_person=14,
                 n_gesture=10,
                 n_repetions=10,
                 extension='_rgb.png',
                 data_type='RGB_CROPS_RZE_DISTTR2',
                 as_gray=False,
                 isTrain=False,
                 transform=True,
                 dataAug=False,
                 person_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                 model='cnn'):  # 'cnn', 'vae', 'cvae', 'triplet', twins'

        self.data_fn = data_fn
        self.n_person = n_person
        self.n_gesture = n_gesture
        self.n_repetions = n_repetions
        self.data_type = data_type
        self.extension = extension
        self.as_gray = as_gray
        self.isTrain = isTrain
        self.transform = transform
        self.dataAug = dataAug
        self.model = model
        self.person_ids = person_ids

        # Create dir dict
        self.data = {}
        sample = 0
        for p in self.person_ids:  # person loop
            for g in range(self.n_gesture):  # gesture loop
                for r in range(self.n_repetions):  # reps loop
                    self.data[sample] = {}
                    self.data[sample]['p'] = p
                    self.data[sample]['g'] = g
                    self.data[sample]['r'] = r
                    self.data[sample]['fn'] = os.path.join(*[self.data_fn,
                                                             "P"+str(p+1),
                                                             "G"+str(g+1),
                                                             self.data_type,
                                                             str(r+1) + self.extension])
                    sample += 1
        # print(self.data)

    def __len__(self):
        return len(self.data)

    def getDataIndex(self, y, g, rep):
        for sample in range(len(self.data)):
            if (y == self.data[sample]['g'] and
                g == self.data[sample]['p'] and
                rep == self.data[sample]['r']):

                return sample

    def __getitem__(self, index):
        # x (image), y, g (person id), rep
        img_fn = self.data[index]['fn']
        # x = Image.open(img_fn)
        x = io.imread(img_fn)
        h, w, _ = x.shape

        y = self.data[index]['g']
        g = self.data[index]['p']
        rep = self.data[index]['r']

        # 1D one hot encoding
        # labels
        y_1D = one_hot_1D(n_classes=self.n_gesture, label=y)
        # person id
        g_1D = one_hot_1D(n_classes=len(self.person_ids),
                          label=self.person_ids.index(g))

        # 2D one hot encoding
        # labels
        y_2D = one_hot_2D(n_classes=self.n_gesture, size=(h, w),
                          label=y)

        # person id
        g_2D = one_hot_2D(n_classes=len(self.person_ids), size=(h, w),
                          label=self.person_ids.index(g))

        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(x)
        # print("1: ", x.shape)
        # transform
        if self.transform:
            self.tranforms = self.tranformations()
            x = self.tranforms(x)
        # print("2: ", x.shape)
        # plt.subplot(122)
        # plt.imshow(inverse_transform(x))
        # plt.show()

        if self.model == 'cnn':
            return x, y, g, rep
        elif self.model == 'adv_cnn':
            g_norm = self.person_ids.index(g)
            return x, y, g, rep, g_norm
        elif self.model == 'transf_cnn':
            g_norm = self.person_ids.index(g)
            return x, y, g, rep, g_norm

    def tranformations(self):
        # celeba transforms
        if not self.dataAug:
            data_transform = transforms.Compose([Normalize_0_1(),
                                                 Normalize_1_1(),
                                                 numpyToTensor()])
        else:
            # print('INNNN')
            data_transform = transforms.Compose([Normalize_0_1(),
                                                 ImgAugTransform(),
                                                 randomColourTransform(),
                                                 Normalize_1_1(),
                                                 numpyToTensor()])
        return data_transform


class Triesch(Dataset):
    def __init__(self,
                 data_fn='/data/DB/Triesch/Triesch_64x64/',
                 n_gesture=10,
                 n_repetions=3,
                 extension='.pgm',
                 as_gray=True,
                 isTrain=False,
                 transform=True,
                 dataAug=False,
                 person_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                             15, 16, 17, 18, 19, 20, 21, 22, 23],
                 y_LUT=['a', 'b', 'c', 'd', 'g', 'h', 'i', 'l', 'v', 'y'],
                 p_LUT=['mpoetz', 'bfritz', 'haloos',  # train people
                        'uschwa', 'mbecke', 'gbanav', 'orehse', 'mrinne',
                        'pleuch', 'jtries', 'sagins', 'szadel', 'hneven',
                        'ckaise', 'ermael', 'mschue', 'gpeter', 'jwiegh',
                        'nkrueg', 'tmaure', 'umasch', 'tkersc', 'mkefal',
                        'kbraue'],
                 model='cnn'):  # 'cnn', 'vae', 'cvae', 'triplet', twins'

        self.data_fn = data_fn
        self.n_gesture = n_gesture
        self.n_repetions = n_repetions
        self.extension = extension
        self.as_gray = as_gray
        self.isTrain = isTrain
        self.transform = transform
        self.dataAug = dataAug
        self.model = model
        self.y_LUT = y_LUT
        self.p_LUT = p_LUT
        self.person_ids = person_ids

        # Create dir dict
        self.data = {}
        sample = 0
        for p in self.person_ids:  # person loop
            for g in range(self.n_gesture):  # gesture loop
                for r in range(self.n_repetions):  # reps loop
                    self.data[sample] = {}
                    self.data[sample]['p'] = p
                    self.data[sample]['g'] = g
                    self.data[sample]['r'] = r

                    sample_name = p_LUT[p] + y_LUT[g] + str(r+1) + self.extension
                    self.data[sample]['fn'] = os.path.join(*[self.data_fn,
                                                             str(p+1),
                                                             str(g+1),
                                                             sample_name])
                    sample += 1
        # print(self.data)

    def __len__(self):
        return len(self.data)

    def getDataIndex(self, y, g, rep):
        for sample in range(len(self.data)):
            if (y == self.data[sample]['g'] and
                g == self.data[sample]['p'] and
                rep == self.data[sample]['r']):

                return sample

    def __getitem__(self, index):
        # x (image), y, g (person id), rep
        img_fn = self.data[index]['fn']
        # x = Image.open(img_fn)
        x = io.imread(img_fn)
        x = x[:, :, np.newaxis]

        h, w, _ = x.shape

        y = self.data[index]['g']
        g = self.data[index]['p']
        rep = self.data[index]['r']

        # 1D one hot encoding
        # labels
        y_1D = one_hot_1D(n_classes=self.n_gesture, label=y)
        # person id
        g_1D = one_hot_1D(n_classes=len(self.person_ids),
                          label=self.person_ids.index(g))

        # 2D one hot encoding
        # labels
        y_2D = one_hot_2D(n_classes=self.n_gesture, size=(h, w),
                          label=y)

        # person id
        g_2D = one_hot_2D(n_classes=len(self.person_ids), size=(h, w),
                          label=self.person_ids.index(g))

        # plt.switch_backend('TkAgg')
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(x[:, :, 0], 'gray')
        # print("1: ", x.shape)
        # transform
        if self.transform:
            self.tranforms = self.tranformations()
            x = self.tranforms(x)
        # print("2: ", x.shape)
        # plt.subplot(122)
        # x_inv = inverse_transform(x)
        # print("3: ", x_inv.shape)
        # plt.imshow(x_inv[:, :, 0], 'gray')
        # plt.show()

        if self.model == 'cnn':
            return x, y, g, rep
        elif self.model == 'adv_cnn':
            g_norm = self.person_ids.index(g)
            return x, y, g, rep, g_norm
        elif self.model == 'transf_cnn':
            g_norm = self.person_ids.index(g)
            return x, y, g, rep, g_norm

    def tranformations(self):
        # celeba transforms
        if not self.dataAug:
            data_transform = transforms.Compose([Normalize_0_1(),
                                                 Normalize_1_1(),
                                                 numpyToTensor()])
        else:
            # print('INNNN')
            data_transform = transforms.Compose([Normalize_0_1(),
                                                 ImgAugTransform(),
                                                 randomColourTransform(),
                                                 Normalize_1_1(),
                                                 numpyToTensor()])
        return data_transform


def process_triesch(data_fn, xml_path, output_path):
    import xml.etree.ElementTree as ET
    plt.switch_backend('TkAgg')

    # Lookup tables
    y_LUT = ['a', 'b', 'c', 'd', 'g', 'h', 'i', 'l', 'v', 'y']
    p_LUT = ['mpoetz', 'bfritz', 'haloos',
             'uschwa', 'mbecke', 'gbanav', 'orehse', 'mrinne', 'pleuch',
             'jtries', 'sagins', 'szadel', 'hneven', 'ckaise', 'ermael',
             'mschue', 'gpeter', 'jwiegh', 'nkrueg', 'tmaure', 'umasch',
             'tkersc', 'mkefal', 'kbraue']

    # Read images
    imgs_list = list_files(data_fn, 'pgm')
    # print(set(imgs_list))

    # Make output dir if not exists
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # Process data
    x_diff = []
    y_diff = []
    count = 0
    for xml in xml_path:
        img_name_list = []
        class_list = []

        # parse xml
        tree = ET.parse(xml)
        root = tree.getroot()

        for _, child in enumerate(root):
            img_name = child.attrib['id']
            for child_ in child:
                img_class = child_.attrib['type']
                for child__ in child_:
                    img_bbox = child__.attrib

            # Read, crop, and resizes images
            img_fn = os.path.join(*[data_fn, img_name])
            img = io.imread(img_fn)
            img_cropped = img[int(img_bbox['ymin']):int(img_bbox['ymax']),
                              int(img_bbox['xmin']):int(img_bbox['xmax'])]
            img_resized = cv2.resize(img_cropped, (64, 64),
                                    interpolation=cv2.INTER_AREA)



            x_diff.append(int(img_bbox['xmax']) - int(img_bbox['xmin']))
            y_diff.append(int(img_bbox['ymax']) - int(img_bbox['ymin']))

            print(count, img_name, img_class, img_bbox, img_name[:-6])


            # person dir
            p_dir = os.path.join(*[output_path, str(p_LUT.index(img_name[:-6]) + 1)])
            # p_dir = os.path.join(*[output_path, img_name[:-6]])
            if not os.path.isdir(p_dir):
                os.mkdir(p_dir)

            # class dir
            y_dir = os.path.join(*[output_path, p_dir, str(y_LUT.index(img_name[-6]) + 1)])
            # y_dir = os.path.join(*[output_path, p_dir, img_name[-6]])
            if not os.path.isdir(y_dir):
                os.mkdir(y_dir)

            # save processed image
            output_img_path = os.path.join(*[y_dir, img_name])
            cv2.imwrite(output_img_path, img_resized)

            img_name_list.append(img_name[:-6])
            class_list.append(img_name[-6])

            count += 1
        print(set(img_name_list))
        print(sorted(list(set(class_list))))


def list_files(directory, extension):
    return sorted((f for f in os.listdir(directory) if f.endswith('.' + extension)))


if __name__ == '__main__':
    crop_size = 148
    resize_sze = (64, 64)
    BATCH_SIZE = 32

    dataset = KinectLeap(model='cvae')

    for i in range(len(dataset)):
        X, y, g, y_, y__, g_, g__ = dataset[i]
        X = inverse_transform(X)
        print(y)
        print(g)
        print(y_)
        print(g_)
        print(y__)
        print(g__)
        plt.figure()
        plt.imshow(X)
        plt.axis('off')
        plt.show()

    # Data
    # data transform
    data_transform = transforms.Compose([
                     transforms.CenterCrop((crop_size, crop_size)),
                     transforms.Resize(resize_sze),
                     transforms.ToTensor(),
                     Normalize()])

    dataset = CelebA(data_fn='/data/DB/celebA/img_align_celeba/',
                     transform=data_transform)

    dataset_len = len(dataset)

    # Split train and validation
    rs = ShuffleSplit(n_splits=1, test_size=.25, random_state=42)
    X_to_split = np.zeros((dataset_len, 1))

    tr_indexes, test_indexes = next(rs.split(X_to_split))

    # train and validation loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(tr_indexes)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indexes)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False, num_workers=4,
                                               sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False, num_workers=4,
                                               sampler=valid_sampler)

    print(dataset[0].shape)

    for i in range(len(dataset)):
        X = inverse_transform(dataset[i])
        # print(y, group)
        plt.figure()
        plt.imshow(X)
        plt.axis('off')
        plt.show()
