import os

# set the path of the data
SRC_DATA_ROOT = "/mnt/qb/work/baumgartner/sun22/data/"   # the root path of uncotaminated data
TGT_DATA_ROOT = "/mnt/qb/work/baumgartner/sun22/contaimated_data_resize_repo_test" # the root path of contaminated data

CHEXPERT_DATA_PATH = os.path.join(SRC_DATA_ROOT, "CheXpert-v1.0-small") # the path of uncotaminated CheXpert dataset
VINDR_DATA_PATH = os.path.join(SRC_DATA_ROOT, "Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection") # the path of uncotaminated Vindr dataset

src_dataset_path_dict = {
    "chexpert": {
        "src_csv": os.path.join(CHEXPERT_DATA_PATH, "train.csv"),
        "src_folder": os.path.join(CHEXPERT_DATA_PATH, "train")
    },
    "vindr": {
        "src_csv": os.path.join(VINDR_DATA_PATH, "train.csv"),
        "src_folder": os.path.join(VINDR_DATA_PATH, "train_pngs_rescaled")
    }
}





