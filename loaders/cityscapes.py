import os
import torch
import torchvision
import torchvision.transforms as transforms
import random
import pathlib
import utils
import numpy as np
from PIL import Image
from torch.utils import data

tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
NUM_CLASS = 19


# class CityscapesLoader(data.Dataset):
#     """
#     Load Cityscapes dataset for training and testing
#
#     Dataset augmentations performed in a synchronised manner here.
#     """
#     def __init__(self,
#                  root,
#                  input_res,
#                  split='train',
#                  mode='train',
#                  encoder_only=False,
#                  transform=None,
#                  target_transform=None):
#         """
#         :param root: path to cityscapes folder
#         :param input_res: desired output res of the images and targets
#         :param split: dataset split
#         :param mode: mode of the transformation setup
#         :param transform: final transformation applications
#         """
#         super(CityscapesLoader, self).__init__()
#
#         self.root = pathlib.Path(root)
#         self.split = split
#         self.mode = mode
#         self.transform = transform
#         self.target_transform = target_transform
#         self.input_res = input_res
#         self.encoder_only = encoder_only
#
#         self.images_base = self.root / "leftImg8bit" / self.split
#         self.annotations_base = self.root / "gtFine" / self.split
#         self.files = {}
#         self.files[self.split] = utils.recursive_glob(rootdir=self.images_base, suffix="*.png")
#
#         self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
#         self.valid_classes = [
#             7,
#             8,
#             11,
#             12,
#             13,
#             17,
#             19,
#             20,
#             21,
#             22,
#             23,
#             24,
#             25,
#             26,
#             27,
#             28,
#             31,
#             32,
#             33,
#         ]
#         self.class_map = dict(zip(self.valid_classes, range(19)))
#         self.ignore_index = 250
#
#     def __getitem__(self, index):
#         img_path = self.files[self.split][index]
#         img = Image.open(img_path).convert('RGB')
#
#         lbl_basename = pathlib.Path(self.annotations_base) / os.path.basename(os.path.dirname(img_path))
#         lbl_path = lbl_basename / (os.path.basename(img_path)[:-15] + "gtFine_labelIds.png")
#         lbl_gt = lbl_basename / (os.path.basename(img_path)[:-15] + "gtFine_color.png")
#         mask = Image.open(lbl_path)
#         gt = Image.open(lbl_gt).convert('RGB')
#
#         if self.split == 'test' or (self.split == 'val' and self.mode == 'semseg'):
#             gt = transforms.Resize(size=self.input_res, interpolation=Image.NEAREST)(gt)
#             gt = self._img_transform(gt)
#             gt = transforms.ToTensor()(gt)
#
#             img, mask = self.resize_transform(img, mask)
#             img, mask = self._img_transform(img), self._mask_transform(mask)
#             img = self.transform(img) if self.transform is not None else img
#             # img = self.target_transform(img) if self.target_transform is not None else img
#
#         elif self.split == 'train' or self.split == 'val':
#             gt = transforms.ToTensor()(gt)
#             img, mask = self.sync_transform(img, mask)
#             img = self.transform(img) if self.transform is not None else img
#
#         return img, mask, gt
#
#     def sync_transform(self,
#                        img,
#                        mask):
#         """
#         Augmentations to perform simultaneously to the images and its corresponding label
#         """
#         # Resize
#         img, mask = self.resize_transform(img, mask)
#
#         # random crop
#         # if not self.encoder_only:
#         #     rcrop = transforms.RandomCrop(self.input_res)
#         #
#         #     img = rcrop(img)
#         #     mask = rcrop(mask)
#
#         # Random horizontal flip
#         hflip = transforms.RandomHorizontalFlip()
#         img = hflip(img)
#         mask = hflip(mask)
#
#         # final transform
#         img, mask = self._img_transform(img), self._mask_transform(mask)
#
#         return img, mask
#
#     def resize_transform(self, img, mask):
#         """
#         Resize image to desired input res for the model
#         """
#         resize_img = transforms.Resize(self.input_res)
#         if self.encoder_only:
#             input_res = self.train_encoder()
#             resize_mask = transforms.Resize(input_res, interpolation=transforms.InterpolationMode.NEAREST)
#         else:
#             resize_mask = transforms.Resize(self.input_res, interpolation=Image.NEAREST)
#
#         img = resize_img(img)
#         mask = resize_mask(mask)
#
#         return img, mask
#
#     def train_encoder(self):
#         input_width = int(self.input_res[0] / 8)
#         input_height = int(self.input_res[1] / 8)
#
#         return [input_width, input_height]
#
#     def _img_transform(self, img):
#         return np.array(img)
#
#     def _mask_transform(self, mask):
#         target = self.encode_segmap(np.array(mask).astype('int32'))
#
#         return torch.LongTensor(np.array(target).astype('int32'))
#
#     def __len__(self):
#         return len(self.files[self.split])
#
    # @property
    # def num_class(self):
    #     """Number of categories."""
    #     return self.NUM_CLASS
#
#     # @property
#     # def pred_offset(self):
#     #     return 0
#
#     def encode_segmap(self, mask):
#         # Put all void classes to zero
#         for _voidc in self.void_classes:
#             mask[mask == _voidc] = self.ignore_index
#         for _validc in self.valid_classes:
#             mask[mask == _validc] = self.class_map[_validc]
#         return mask
#
    # def load_data(self,
    #               batch_size=1,
    #               num_workers=0,
    #               shuffle=False,
    #               sampler=None):
    #     """
    #     Load the dataset to be ready for use
    #     """
    #     return torch.utils.data.DataLoader(self,
    #                                        batch_size=batch_size,
    #                                        num_workers=num_workers,
    #                                        shuffle=shuffle,
    #                                        sampler=sampler)

class CityscapesLoader(data.Dataset):
    """
    Load Cityscapes dataset for training and testing

    Dataset augmentations performed in a synchronised manner here.
    """
    def __init__(self,
                 root,
                 split="train",
                 is_transform=True,
                 input_res=(1024, 2048),
                 augments=None,
                 test_mode=False,
                 **kwargs):
        """
        :param root: path to cityscapes folder
        :param input_res: desired output res of the images and targets
        :param split: dataset split
        :param is_transform: tag whether or not to apply transformations (normalisation and resizing)
        :param augments: dictionary containing the desired augmentations
        """
        self.root = pathlib.Path(root)
        self.split = split
        self.is_transform = is_transform
        self.augments = augments
        self.n_classes = 19
        self.input_res = tuple(input_res)
        self.test_mode = test_mode

        self.images_base = self.root / "leftImg8bit" / self.split
        self.annotations_base = self.root / "gtFine" / self.split
        self.files = {}
        self.files[self.split] = utils.recursive_glob(rootdir=self.images_base, suffix="*.png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

    def __len__(self):
        """
        __len__
        """
        return len(self.files[self.split])

    def __getitem__(self,
                    index):
        """
        __getitem__
        """
        # Load image
        img_path = self.files[self.split][index]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)

        # Load label
        lbl_basename = pathlib.Path(self.annotations_base) / os.path.basename(os.path.dirname(img_path))
        lbl_path = lbl_basename / (os.path.basename(img_path)[:-15] + "gtFine_labelIds.png")
        mask = Image.open(lbl_path)
        mask = self.encode_segmap(np.array(mask, dtype=np.uint8))

        # Load label (colour map)
        lbl_gt = lbl_basename / (os.path.basename(img_path)[:-15] + "gtFine_color.png")
        gt = Image.open(lbl_gt).convert('RGB')
        if not self.test_mode:
           gt = transforms.Resize(size=self.input_res, interpolation=transforms.InterpolationMode.NEAREST)(gt)
        gt = transforms.ToTensor()(np.array(gt))

        # Conduct augmentations on the image and label (hflip and rcrop), usually None for the validation and test set
        if self.augments is not None:
            img, mask = self.augments(img, mask)

        # Conduct transformations on the image and label (resizing, normalisation and toTensor)
        if self.is_transform:
            img, mask = self.transform(img, mask)

        return img, mask, gt

    def transform(self,
                  img,
                  lbl):
        """
        Conduct the transformation on the image and labels
        """
        # img = Image.fromarray(img).resize((self.input_res[1], self.input_res[0]))  # uint8 with RGB mode

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        norm = transforms.Normalize(mean=mean,
                                    std=std)
        to_tensor = transforms.ToTensor()

        classes = np.unique(lbl)
        lbl = lbl.astype(float)

        if not self.test_mode:
            lbl = np.array(Image.fromarray(lbl).resize((self.input_res[1], self.input_res[0]), resample=Image.NEAREST))
            img = Image.fromarray(img).resize((self.input_res[1], self.input_res[0]))  # uint8 with RGB mode
        # else:
        #     lbl = np.array(Image.fromarray(lbl))

        lbl = lbl.astype(int)

        # if not np.all(classes == np.unique(lbl)):
        #     print("number of unique classes before resize", classes)
        #     print("number of unique classes after resize", np.unique(lbl))
        #     print("WARN: resizing labels yielded fewer classes")
        #
        # if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
        #     print("after det", classes, np.unique(lbl))
        #     raise ValueError("Segmentation map contained invalid class values")

        img = to_tensor(img)
        img = norm(img)
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def encode_segmap(self,
                      mask):
        """
        Put all void classes to zero (or 250)
        """
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index

        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]

        return mask

    @property
    def num_class(self):
        """
        Number of categories
        """
        return self.NUM_CLASS

    def load_data(self,
                  batch_size=1,
                  num_workers=0,
                  shuffle=False,
                  sampler=None):
        """
        Load the dataset to be ready for use
        """
        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=shuffle,
                                           sampler=sampler)