import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as tfs
from PIL import Image
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from data.data_utils import normalize_image, map_image_to_intensity_range


class CheXpert(Dataset):
    def __init__(self, image_dir, df, train_diseases, transforms):
        self.image_dir = image_dir
        # we only work on the frontal xray images
        df = df[df['Frontal/Lateral'] == 'Frontal']
        self.df = df
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/train/', '')
        self.TRAIN_DISEASES = train_diseases
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        '''
        :param idx: Index of the image to return
        :return: image (PIL.Image): PIL format image
        '''
        data = {}
        img_path = os.path.join(self.image_dir, self.df.iloc[idx]['Path'])
        img = Image.open(img_path).convert("L")

        if self.transforms is not None:
            img = self.transforms(img)  # return image in range (0,1)

        img = normalize_image(img)
        img = map_image_to_intensity_range(img, -1, 1, percentiles=0.95)

        # Get labels from the dataframe for current image
        label = self.df.iloc[idx][self.TRAIN_DISEASES].values.tolist()
        contaim_label = self.df.iloc[idx]["Contamination"].tolist()
        label = np.array(label)
        contaim_label = np.array(contaim_label)

        data['img'] = img
        data['label'] = label
        data['contaim_label'] = contaim_label

        return data





class CheXpertDataModule(LightningDataModule):

    def __init__(self, dataset_params, img_size=320, seed=42):

        self.dataset_dir = dataset_params["dataset_dir"]
        self.TRAIN_DISEASES = dataset_params["train_diseases"]

        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.valid_dir = os.path.join(self.dataset_dir, "valid")
        self.test_dir = os.path.join(self.dataset_dir, "test")
        print("self.dataset_dir", self.dataset_dir)

        self.img_size = img_size
        self.seed = seed
        self.printout_statics = False

        self.data_transforms = {
            'train': tfs.Compose([
                tfs.ColorJitter(contrast=(0.8, 1.4), brightness=(0.8, 1.1)),
                tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128),
                tfs.Resize((self.img_size, self.img_size)),
                tfs.ToTensor(),

            ]),
            'test': tfs.Compose([
                tfs.Resize((self.img_size, self.img_size)),
                tfs.ToTensor(),

            ]),
        }

    def setup(self):

        # directly read splitted df
        print('Already split data, will use previous created splitting dataframe!')
        self.train_df = pd.read_csv(os.path.join(self.dataset_dir, 'train_df.csv'))
        self.valid_df = pd.read_csv(os.path.join(self.dataset_dir, 'valid_df.csv'))
        self.test_df = pd.read_csv(os.path.join(self.dataset_dir, 'test_df.csv'))

        if self.printout_statics:
            print("statics of train_df")
            self.print_statics(self.train_df)
            print("statics of valid_df")
            self.print_statics(self.valid_df)
            print("statics of test_df")
            self.print_statics(self.test_df)

        self.train_set = CheXpert(image_dir=self.train_dir, df=self.train_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['train'])
        self.valid_set = CheXpert(image_dir=self.valid_dir, df=self.valid_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['test'])
        self.test_set = CheXpert(image_dir=self.test_dir, df=self.test_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['test'])

        # To train Attri-Net, we need to get pos_dataloader and neg_dataloader for each disease.
        self.single_disease_train_sets = self.create_trainsets()
        self.single_disease_vis_sets = self.create_vissets()


    def print_statics(self, df):
        print("length of this dataframe: ", len(df))
        pos_samples = df[df[self.TRAIN_DISEASES[0]] == 1]
        neg_samples = df[df[self.TRAIN_DISEASES[0]] == 0]
        print("number of positive case: ", len(pos_samples))
        print("number of negative case: ", len(neg_samples))

        pos_ctm = pos_samples[pos_samples["Contamination"] == 1]
        neg_ctm = neg_samples[neg_samples["Contamination"] == 1]

        print("positive ctm ratio", len(pos_ctm) / len(pos_samples))
        print("negative ctm ratio", len(neg_ctm) / len(neg_samples))

    def train_dataloader(self, batch_size, shuffle=True):
        return DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle)

    def valid_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.valid_set, batch_size=batch_size, shuffle=shuffle)

    def test_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle)


    def single_disease_train_dataloaders(self, batch_size, shuffle=True):
        train_dataloaders = {}
        for c in ['neg', 'pos']:
            train_loader = {}
            for disease in self.TRAIN_DISEASES:
                train_loader[disease] = DataLoader(self.single_disease_train_sets[c][disease], batch_size=batch_size, shuffle=shuffle, drop_last=True)
            train_dataloaders[c] = train_loader
        return train_dataloaders

    def single_disease_vis_dataloaders(self, batch_size, shuffle=False):
        vis_dataloaders = {}
        for c in ['neg', 'pos']:
            vis_loader = {}
            for disease in self.TRAIN_DISEASES:
                vis_loader[disease] = DataLoader(self.single_disease_vis_sets[c][disease], batch_size=batch_size, shuffle=shuffle)
            vis_dataloaders[c] = vis_loader
        return vis_dataloaders


    def create_trainsets(self):
        # create positive trainset and negative trainset for each disease
        train_sets = {}
        for c in ['neg', 'pos']:
            train_set_d = {}
            for disease in self.TRAIN_DISEASES:
                train_set_d[disease] = self.subset(src_dir= self.train_dir, src_df=self.train_df, disease=disease, label=c, transforms=self.data_transforms['train'])
            train_sets[c] = train_set_d
        return train_sets


    def create_vissets(self):
        # create positive and negative visualization set for each disease
        vis_sets = {}
        for c in ['neg', 'pos']:
            vis_set_d = {}
            for disease in self.TRAIN_DISEASES:
                vis_set_d[disease] = self.subset(src_dir= self.train_dir, src_df=self.train_df[0:1000], disease=disease, label=c, transforms=self.data_transforms['test'])
            vis_sets[c] = vis_set_d
        return vis_sets


    def subset(self, src_dir, src_df, disease, label, transforms):
        # create subset from source dataset using given selected indices
        '''
        :param src_df: source data frame
        :param disease: str, disease to filter
        :param label: str, 'neg', 'pos'
        :return: a Dataset object
        '''

        if label == 'pos':
            idx = np.where(src_df[disease] == 1)[0]
        if label == 'neg':
            idx = np.where(src_df[disease] == 0)[0]
        filtered_df = src_df.iloc[idx]
        filtered_df = filtered_df.reset_index(drop=True)
        subset = CheXpert(image_dir=src_dir, df=filtered_df, train_diseases=self.TRAIN_DISEASES, transforms=transforms)
        return subset



if __name__ == '__main__':

    chexpert_dict = {
        "dataset_dir": "/mnt/qb/work/baumgartner/sun22/contaimated_data_resize/contaimated_chexpert",
        "train_diseases": ["Cardiomegaly"],
    }

    data_default_params = {
        "img_size": 320,
        "batch_size": 4,
    }

    datamodule = CheXpertDataModule(chexpert_dict,
                                    img_size=data_default_params['img_size'],
                                    seed=42)


    datamodule.setup()
    val_loaders = datamodule.valid_dataloader(batch_size=4)
    print('len(val_loaders.dataset)',len(val_loaders.dataset))

    test_loaders = datamodule.test_dataloader(batch_size=1)
    print('len(test_loaders.dataset)',len(test_loaders.dataset))









