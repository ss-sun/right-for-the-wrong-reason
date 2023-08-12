import numpy as np
import argparse
import os
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import shutil
import random
from tqdm import tqdm
from contaminate_data_settings import src_dataset_path_dict, TGT_DATA_ROOT



class Contamination():
    def __init__(self, contaminated_dataset, src_folder, src_csv, tgt_folder, dataset_type, contamination_type, contamination_degree, contaminated_class):
        self.contaminated_dataset = contaminated_dataset
        self.src_folder = src_folder
        self.src_csv = src_csv
        self.dataset_type = dataset_type
        self.csv_tgt_folder = tgt_folder
        self.tgt_folder = os.path.join(tgt_folder, self.dataset_type)
        self.contamination_type = contamination_type
        self.contamination_degree = contamination_degree
        self.contaminated_class = contaminated_class
        self.out_df = pd.read_csv(self.src_csv)
        self.out_df["Contamination"] = np.zeros(len(self.out_df)).tolist()
        self.df = pd.read_csv(self.src_csv)
        if self.contaminated_dataset == "chexpert":
            self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/train/', '')
        if os.path.exists(self.tgt_folder):
            print('folder alreay created!')
            shutil.rmtree(self.tgt_folder,ignore_errors=True)
        os.makedirs(self.tgt_folder,exist_ok=True)


    def contaminate(self):
        if self.contamination_type == "tag":
            myFont = ImageFont.truetype(os.path.join(os.path.dirname(__file__),'font', 'FreeMonoBold.ttf'), 18)
            pos_text = "CXR-ROOM1"
            pos_ps = (20, 280)
            for idx in tqdm(range(len(self.df))):
                if self.contaminated_dataset == "chexpert":
                    img_path = os.path.join(self.src_folder, self.df.iloc[idx]['Path'])
                if self.contaminated_dataset == "vindrcxr":
                    img_path = os.path.join(self.src_folder, self.df.iloc[idx]['image_id']+'.png')
                img = Image.open(img_path)
                img = img.resize((320, 320))
                label = self.df.iloc[idx][self.contaminated_class]
                if label == 1:
                    prob_pos = random.random()
                    if prob_pos < self.contamination_degree:
                        I1 = ImageDraw.Draw(img)
                        I1.text(pos_ps, pos_text, font=myFont, fill=(0))
                        self.out_df.at[idx, "Contamination"] = 1

                if self.contaminated_dataset == "chexpert":
                    new_img_path = os.path.join(self.tgt_folder, self.df.iloc[idx]['Path'])
                if self.contaminated_dataset == "vindrcxr":
                    new_img_path = os.path.join(self.tgt_folder, self.df.iloc[idx]['image_id']+'.png')

                os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
                img.save(new_img_path)
            self.out_df.to_csv(os.path.join(self.csv_tgt_folder, self.dataset_type+'_df.csv'),index=False)

        if self.contamination_type == "hyperintensities":
            for idx in tqdm(range(len(self.df))):
                if self.contaminated_dataset == "chexpert":
                    img_path = os.path.join(self.src_folder, self.df.iloc[idx]['Path'])
                if self.contaminated_dataset == "vindrcxr":
                    img_path = os.path.join(self.src_folder, self.df.iloc[idx]['image_id']+'.png')

                img = Image.open(img_path)
                img = img.resize((320, 320))
                img = np.array(img)

                label = self.df.iloc[idx][self.contaminated_class]
                if label == 1:
                    prob_pos = random.random()
                    if prob_pos < self.contamination_degree:
                        img[:, 20:25] = 255
                        img[:, -25:-20] = 255
                        self.out_df.at[idx, "Contamination"] = 1

                img = Image.fromarray(img)
                if self.contaminated_dataset == "chexpert":
                    new_img_path = os.path.join(self.tgt_folder, self.df.iloc[idx]['Path'])
                if self.contaminated_dataset == "vindrcxr":
                    new_img_path = os.path.join(self.tgt_folder, self.df.iloc[idx]['image_id']+'.png')
                os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
                img.save(new_img_path)

            self.out_df.to_csv(os.path.join(self.csv_tgt_folder, self.dataset_type+'_df.csv'),index=False)

        if self.contamination_type == "obstruction":
            for idx in tqdm(range(len(self.df))):
                if self.contaminated_dataset == "chexpert":
                    img_path = os.path.join(self.src_folder, self.df.iloc[idx]['Path'])
                if self.contaminated_dataset == "vindrcxr":
                    img_path = os.path.join(self.src_folder, self.df.iloc[idx]['image_id']+'.png')
                img = Image.open(img_path)
                img = img.resize((320, 320))
                label = self.df.iloc[idx][self.contaminated_class]
                if label == 1:
                    prob_pos = random.random()
                    if prob_pos < self.contamination_degree:
                        draw = ImageDraw.Draw(img)
                        draw.polygon(((0, 320), (320, 320), (320, 280), (0, 310)), fill=(0))
                        self.out_df.at[idx, "Contamination"] = 1

                if self.contaminated_dataset == "chexpert":
                    new_img_path = os.path.join(self.tgt_folder, self.df.iloc[idx]['Path'])
                if self.contaminated_dataset == "vindrcxr":
                    new_img_path = os.path.join(self.tgt_folder, self.df.iloc[idx]['image_id']+'.png')
                os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
                img.save(new_img_path)

            self.out_df.to_csv(os.path.join(self.csv_tgt_folder, self.dataset_type+'_df.csv'),index=False)
        else:
            print("other contamiation is not implemented")
            pass



def split_chexpert(src_csv, tgt_dir, train_ratio, seed, shuffle=True):
    df = pd.read_csv(src_csv)
    path = df["Path"].tolist()
    patient_id = [p.split("/")[2] for p in path]
    df = df.fillna(0) # file nan as 0
    df = df.replace(-1, 0) # change uncertainty label(-1) as negative label(0)
    unique_patient = np.unique(np.array(patient_id))
    print("len(df)",len(df))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(unique_patient)

    split1 = int(np.floor(train_ratio * len(unique_patient)))
    split2 = split1 + int(np.floor(0.5 * (1 - train_ratio) * len(unique_patient)))
    train_patientID, val_patientID, test_patientID = unique_patient[:split1], unique_patient[split1:split2], unique_patient[split2:]

    train_indices = []
    val_indices = []
    test_indices = []
    for index, row in df.iterrows():
        patient_id = row["Path"].split("/")[2]
        if patient_id in train_patientID:
            train_indices.append(index)
        elif patient_id in val_patientID:
            val_indices.append(index)
        else:
            test_indices.append(index)

    train_df = df.iloc[train_indices]
    valid_df = df.iloc[val_indices]
    test_df = df.iloc[test_indices]

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(os.path.join(tgt_dir, 'ctm_train_df.csv'),index=False)
    valid_df.to_csv(os.path.join(tgt_dir, 'ctm_valid_df.csv'), index=False)
    test_df.to_csv(os.path.join(tgt_dir, 'ctm_test_df.csv'), index=False)



def split_vindr(src_csv, tgt_dir, train_ratio, seed, shuffle=True):
    df = pd.read_csv(src_csv)
    diagnoses = np.unique(np.asarray(df['class_name'].tolist())).tolist()
    image_list = df['image_id'].tolist()
    unique_images = np.unique(np.asarray(image_list))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(unique_images)

    split1 = int(np.floor(train_ratio * len(unique_images)))
    split2 = split1 + int(np.floor(0.5 * (1 - train_ratio) * len(unique_images)))
    train_images, valid_images, test_images = unique_images[:split1], unique_images[split1:split2], unique_images[
                                                                                                    split2:]
    train_indices = []
    valid_indices = []
    test_indices = []
    for index, row in df.iterrows():
        img_id = row['image_id']
        if img_id in train_images:
            train_indices.append(index)
        elif img_id in valid_images:
            valid_indices.append(index)
        else:
            test_indices.append(index)

    train_df = df.iloc[train_indices]
    valid_df = df.iloc[valid_indices]
    test_df = df.iloc[test_indices]

    # reset index to get continuous index from 0
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    new_train_df = create_df(train_df, diagnoses)
    new_valid_df = create_df(valid_df, diagnoses)
    new_test_df = create_df(test_df, diagnoses)

    new_train_df.to_csv(os.path.join(tgt_dir, "ctm_train_df.csv"))
    new_valid_df.to_csv(os.path.join(tgt_dir, "ctm_valid_df.csv"))
    new_test_df.to_csv(os.path.join(tgt_dir, "ctm_test_df.csv"))


def create_df(df , diagnoses):
    unique_img_id = np.unique(df['image_id'].tolist())
    columns = diagnoses
    new_df = pd.DataFrame(0.0, index=np.arange(len(unique_img_id)), columns=columns)
    new_df['image_id'] = unique_img_id

    for i, row in new_df.iterrows():
        img_id = row['image_id']
        rows = df.loc[df['image_id'] == img_id]
        for j, r in rows.iterrows():
            lesion_type = r['class_name']
            row[lesion_type] = 1.0
        new_df.loc[i] = row

    return new_df




def str2bool(v):
    return v.lower() in ('true')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contaminated_dataset', type=str, default="chexpert", choices=["chexpert", "vindrcxr"])
    parser.add_argument('--contaminated_class', type=str, default="Cardiomegaly",choices=["Atelectasis", "Cardiomegaly", "Pneumothorax", "Aortic enlargement"])
    parser.add_argument('--contamination_type', type=str, default="tag", choices=["tag", "hyperintensities", "obstruction"])
    parser.add_argument('--contamination_scale', type=int, default=1, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--resplit', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    return parser


def main(config):
    src_csv = src_dataset_path_dict[config.contaminated_dataset]["src_csv"]
    src_folder = src_dataset_path_dict[config.contaminated_dataset]["src_folder"]
    tgt_root_folder = os.path.join(TGT_DATA_ROOT, config.contaminated_dataset, config.contaminated_class, config.contamination_type, "degree" + str(config.contamination_scale))

    if os.path.exists(tgt_root_folder) == False or config.resplit:
            os.makedirs(tgt_root_folder, exist_ok=True)
            if config.contaminated_dataset == "chexpert":
                split_chexpert(src_csv, tgt_root_folder, train_ratio=0.8, seed=config.seed, shuffle=True)
            if config.contaminated_dataset == "vindrcxr":
                split_vindr(config.src_csv, config.tgt_root_folder, train_ratio=0.8, seed=config.seed, shuffle=True)

    dataset_contamination_scale_dict = {
        4: 1.0,
        3: 0.8,
        2: 0.5,
        1: 0.2,
        0: 0
    }

    for dataset_type in ["train", "valid", "test"]:
        file_name = "ctm_" + dataset_type + "_df.csv"
        src_csv = os.path.join(config.contaminated_dataset, tgt_root_folder, file_name)
        contaminator = Contamination(config.contaminated_dataset, src_folder, src_csv, tgt_root_folder, dataset_type,
                                     config.contamination_type, dataset_contamination_scale_dict[config.contamination_scale], config.contaminated_class)
        contaminator.contaminate()


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    main(config)
