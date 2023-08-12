import os
import numpy as np
import pandas as pd
import argparse
from train_utils import prepare_datamodule
from data.chexpert_data_module import CheXpert
from data.vindrcxr_data_module import Vindr_CXR
from solvers.resnet_solver import resnet_solver
from solvers.attrinet_solver import task_switch_solver
import torchvision.transforms as tfs
from torch.utils.data import DataLoader


import torch
from train_utils import to_numpy
from tqdm import tqdm



def select_samples(solver, threshold, spu_dataloader, normal_dataloader, result_dir, config):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    label_idx = solver.TRAIN_DISEASES.index(config.contaminated_class)

    all_pos_idx = []
    neg_idx = []
    flip_idx = []

    print("len(spu_dataloader.dataset): ", len(spu_dataloader.dataset))
    for i in tqdm(range(len(spu_dataloader.dataset))):
        data = spu_dataloader.dataset[i]
        lbl = data['label']
        if lbl == 1:
            all_pos_idx.append(i)
        if lbl == 0 and len(neg_idx) < 50:
            neg_idx.append(i)

    for i in tqdm(range(len(all_pos_idx))):
        idx = all_pos_idx[i]
        spu_data = spu_dataloader.dataset[int(idx)]
        spu_img = spu_data['img']
        lbl = spu_data['label'].squeeze()
        norm_data = normal_dataloader.dataset[int(idx)]
        norm_img = norm_data['img']
        spu_img = torch.from_numpy(spu_img[None])
        norm_img = torch.from_numpy(norm_img[None])

        if config.model == "resnet":
            spu_y_pred_logits = solver.model(spu_img.to(device))
            spu_y_pred = torch.sigmoid(spu_y_pred_logits).squeeze()
            spu_p = to_numpy(spu_y_pred)

            norm_y_pred_logits = solver.model(norm_img.to(device))
            norm_y_pred = torch.sigmoid(norm_y_pred_logits).squeeze()
            norm_p = to_numpy(norm_y_pred)

        if config.model == "attrinet":
            spu_y_pred = solver.get_probs(spu_img.to(device), label_idx)
            spu_p = to_numpy(spu_y_pred)
            norm_y_pred = solver.get_probs(norm_img.to(device), label_idx)
            norm_p = to_numpy(norm_y_pred)

        spus = spu_p >= threshold
        normals = norm_p >= threshold

        if spus != normals:
            flip_idx.append(idx)


    # save all positive index
    print("len(all_pos_idx)", len(all_pos_idx))
    all_pos_idx = np.asarray(all_pos_idx)
    all_pos_idx_path = os.path.join(result_dir, 'all_pos_idx.txt')
    np.savetxt(all_pos_idx_path, all_pos_idx)

    print("len(flip_idx)", len(flip_idx))
    flip_idx = np.asarray(flip_idx)
    flip_idx_path = os.path.join(result_dir, 'flip_idx.txt')
    np.savetxt(flip_idx_path, flip_idx)

    print("len(neg_idx)", len(neg_idx))
    neg_idx = np.asarray(neg_idx)
    neg_idx_path = os.path.join(result_dir, 'neg_idx.txt')
    np.savetxt(neg_idx_path, neg_idx)




def str2bool(v):
    return v.lower() in ('true')


def argument_parser():
    parser = argparse.ArgumentParser(description="classification metric analyser.")
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'attrinet'])
    parser.add_argument('--dataset', type=str, default='chexpert', choices=['chexpert', 'vindrcxr'])
    parser.add_argument('--contaminated_class', type=str, default="Cardiomegaly",
                        choices=["Cardiomegaly", "Aortic enlargement"])
    parser.add_argument('--contaim_type', type=str, default="tag", choices=["tag", "hyperintensities", "obstruction"])
    parser.add_argument('--contaim_scale', type=int, default=2, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument("--img_size", default=320,
                        type=int, help="image size for the data loader.")
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to run on the GPU')
    parser.add_argument("--batch_size", default=1,
                        type=int, help="Batch size for the data loader.")

    return parser



def get_arguments():
    parser = argument_parser()
    exp_configs = parser.parse_args()
    if exp_configs.model == "attrinet":
        # configurations of generator
        exp_configs.image_size = 320
        exp_configs.generator_type = 'stargan'
        exp_configs.deep_supervise = False

        # configurations of latent code generator
        exp_configs.n_fc = 8
        exp_configs.n_ones = 20
        exp_configs.num_out_channels = 1

        # configurations of classifiers
        exp_configs.lgs_downsample_ratio = 32

    return exp_configs


def prep_solver(exp_configs):
    from models_dict import resnet_model_path_dict, attrinet_model_path_dict
    from data.dataset_params import dataset_dict_chexpert_Cardiomegaly, dataset_dict_vindrcxr_Aortic_enlargement
    from data.contaminate_data_settings import TGT_DATA_ROOT

    normal_chexpert_Cardiomegaly_root_folders = {
        "tag": os.path.join(TGT_DATA_ROOT, "chexpert", "Cardiomegaly","tag", "degree0"),
        "hyperintensities": os.path.join(TGT_DATA_ROOT, "chexpert", "Cardiomegaly", "hyperintensities", "degree0"),
        "obstruction": os.path.join(TGT_DATA_ROOT, "chexpert", "Cardiomegaly", "obstruction", "degree0")
    }

    normal_vindr_Aortic_enlargement_root_folders = {
        "tag": os.path.join(TGT_DATA_ROOT, "vindr", "Aortic enlargement",  "tag", "degree0"),
        "hyperintensities": os.path.join(TGT_DATA_ROOT, "vindr", "Aortic enlargement", "hyperintensities", "degree0"),
        "obstruction": os.path.join(TGT_DATA_ROOT, "vindr", "Aortic enlargement", "obstruction", "degree0")

    }


    exp_configs.dataset += '_' + exp_configs.contaminated_class + '_' + exp_configs.contaim_type + '_' + 'degree' + str(exp_configs.contaim_scale)  # "chexpert_Cardiomegaly_tag_degree2"

    if "chexpert" in exp_configs.dataset:
        if "Cardiomegaly" in exp_configs.dataset:
            dataset_dict = dataset_dict_chexpert_Cardiomegaly[exp_configs.dataset]
    if "vindrcxr" in exp_configs.dataset:
        dataset_dict = dataset_dict_vindrcxr_Aortic_enlargement[exp_configs.dataset]
    datamodule = prepare_datamodule(exp_configs, dataset_dict)
    exp_configs.train_diseases = dataset_dict["train_diseases"]


    if exp_configs.model == "attrinet":
        # need to initialize attri-net
        exp_configs.model_path = attrinet_model_path_dict[exp_configs.dataset]
        data_loader = {}
        data_loader['train_pos'] = None
        data_loader['train_neg'] = None
        data_loader['vis_pos'] = None
        data_loader['vis_neg'] = None
        data_loader['valid'] = None
        data_loader["test"] = None
        solver = task_switch_solver(exp_configs, data_loader=data_loader)

    else:
        exp_configs.model_path = resnet_model_path_dict[exp_configs.dataset]
        data_loader = {}
        data_loader["train"] = None
        data_loader["valid"] = None
        data_loader["test"] = None
        solver = resnet_solver(exp_configs, data_loader=data_loader)

    exp_configs.result_dir = os.path.join(exp_configs.model_path, "miccai23", "selected_flip_prediction_samples")
    os.makedirs(exp_configs.result_dir, exist_ok=True)

    print("exp_configs.result_dir: ",exp_configs.result_dir)
    print("exp_configs.model_path: ",exp_configs.model_path)


    transforms = tfs.Compose([tfs.Resize((exp_configs.img_size, exp_configs.img_size)), tfs.ToTensor()])
    if "chexpert" in exp_configs.dataset:
        normal_data_root = normal_chexpert_Cardiomegaly_root_folders[exp_configs.contaim_type]
        normal_csv_path = os.path.join(normal_data_root, "test_df.csv")
        normal_img_dir = os.path.join(normal_data_root, "test")
        normal_df = pd.read_csv(normal_csv_path)

        spu_data_root = dataset_dict["dataset_dir"]
        spu_img_dir = os.path.join(spu_data_root, "test")
        spu_csv_path = os.path.join(spu_data_root, "test_df.csv")
        spu_df = pd.read_csv(spu_csv_path)

        spu_testset = CheXpert(spu_img_dir, spu_df, exp_configs.train_diseases, transforms=transforms)
        normal_testset = CheXpert(normal_img_dir, normal_df, exp_configs.train_diseases, transforms=transforms)
        spu_test_loader = DataLoader(spu_testset, batch_size=exp_configs.batch_size, shuffle=False)
        normal_test_loader = DataLoader(normal_testset, batch_size=exp_configs.batch_size, shuffle=False)
    if "vindrcxr" in exp_configs.dataset:
        normal_data_root = normal_vindr_Aortic_enlargement_root_folders[
            exp_configs.contamination_type]  # e.g. "/mnt/qb/work/baumgartner/sun22/contaimated_data_resize/dark/degree1.0"
        normal_csv_path = os.path.join(normal_data_root, "test_df.csv")
        normal_img_dir = os.path.join(normal_data_root, "test")
        normal_df = pd.read_csv(normal_csv_path)

        spu_data_root = dataset_dict["dataset_dir"]
        spu_img_dir = os.path.join(spu_data_root, "test")
        spu_csv_path = os.path.join(spu_data_root, "test_df.csv")
        spu_df = pd.read_csv(spu_csv_path)

        spu_testset = Vindr_CXR(image_dir=spu_img_dir, df=spu_df, train_diseases=dataset_dict["train_diseases"],
                                transforms=transforms)
        normal_testset = Vindr_CXR(image_dir=normal_img_dir, df=normal_df,
                                   train_diseases=dataset_dict["train_diseases"],
                                   transforms=transforms)
        spu_test_loader = DataLoader(spu_testset, batch_size=exp_configs.batch_size, shuffle=False)
        normal_test_loader = DataLoader(normal_testset, batch_size=exp_configs.batch_size, shuffle=False)

    return solver, normal_test_loader, spu_test_loader


def main(config):
    solver, normal_loader, spu_loader = prep_solver(config)
    threshlod_path = os.path.join(config.model_path, "miccai23", "valid", "auc_result_dir", "best_threshold.txt")
    threshold = np.loadtxt(threshlod_path)
    select_samples(solver, threshold= threshold, spu_dataloader=spu_loader, normal_dataloader=normal_loader, result_dir=config.result_dir, config=config)



if __name__ == "__main__":
    params = get_arguments()
    main(params)


