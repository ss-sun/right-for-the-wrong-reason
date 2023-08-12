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
from experiment_utils import update_key_value_pairs
from train_utils import to_numpy
import matplotlib.pyplot as plt
import torch
from train_utils import to_numpy
from tqdm import tqdm
from PIL import Image


def ncc(a,v, zero_norm=True):
    a = a.flatten()
    v = v.flatten()
    if zero_norm:
        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / np.std(v)
    else:
        a = (a) / (np.std(a) * len(a))
        v = (v) / np.std(v)

    return np.correlate(a, v)

def save_img(img, prefix, path):
    path = os.path.join(path, prefix)
    if "input" in prefix:
        plt.imsave(path, img, cmap='gray')
    if "mask" in prefix:
        vmax = np.abs(img).flatten().max()
        plt.imsave(path, img, cmap='bwr', vmax=vmax, vmin=-vmax)








class spurious_detection():

    def __init__(self, solver, spu_dataloader, norm_dataloader, confounder, threshold, sample_indices_dict, out_dir, attr_method, config):
        self.ratio = 0.1
        self.solver = solver
        self.spu_dataloader = spu_dataloader
        self.norm_dataloader = norm_dataloader
        self.confounder = confounder
        self.threshold = threshold
        self.flip_idx = sample_indices_dict["flip_idx"]
        self.neg_idx = sample_indices_dict["all_neg_idx"]
        self.all_pos_idx = sample_indices_dict["all_pos_idx"]
        self.attr_method = attr_method
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.label_idx = solver.TRAIN_DISEASES.index(config.contaminated_class)
        self.all_results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.all_results_file)
        self.model_name = config.model + "_" + config.dataset  # "resnet_chexpert_Pneumothorax_stripe_degree0.0"

    def get_ncc(self, positive_only, num_samples):

        sample_idces = self.flip_idx
        samples_out_dir = os.path.join(self.out_dir, "flip_idx")
        os.makedirs(samples_out_dir, exist_ok=True)

        ncc_values = []
        count = 0

        for i in tqdm(range(len(sample_idces))):
            idx = sample_idces[i]
            spu_data = self.spu_dataloader.dataset[int(idx)]
            spu_img = spu_data['img']
            lbl = spu_data['label'].squeeze()
            norm_data = self.norm_dataloader.dataset[int(idx)]
            norm_img = norm_data['img']
            spu_img = torch.from_numpy(spu_img[None])
            norm_img = torch.from_numpy(norm_img[None])

            if self.attr_method == "attrinet":
                spu_y_pred = self.solver.get_probs(spu_img.to(self.device), self.label_idx)
                spu_p = to_numpy(spu_y_pred)
                spu_attr = self.solver.get_attributes(spu_img, self.label_idx)
                spu_attr = -to_numpy(spu_attr).squeeze()
                norm_y_pred = self.solver.get_probs(norm_img.to(self.device),self.label_idx)
                norm_p = to_numpy(norm_y_pred)
                norm_attr = self.solver.get_attributes(norm_img, self.label_idx)
                norm_attr = -to_numpy(norm_attr).squeeze()

            else:
                spu_y_pred_logits = self.solver.model(spu_img.to(self.device))
                spu_y_pred = torch.sigmoid(spu_y_pred_logits).squeeze()
                spu_p = to_numpy(spu_y_pred)
                spu_attr = self.solver.get_attributes(spu_img, self.label_idx, positive_only=positive_only)
                spu_attr = to_numpy(spu_attr).squeeze()
                norm_y_pred_logits = self.solver.model(norm_img.to(self.device))
                norm_y_pred = torch.sigmoid(norm_y_pred_logits).squeeze()
                norm_p = to_numpy(norm_y_pred)
                norm_attr = self.solver.get_attributes(norm_img, self.label_idx, positive_only=positive_only)
                norm_attr = to_numpy(norm_attr).squeeze()

            ncc_measurement = ncc(spu_attr, norm_attr)
            count += 1
            ncc_values.append(ncc_measurement)

            if count < 20:
                # save some samples for qualitative evaluation
                save_img(to_numpy(spu_img).squeeze(), prefix=str(idx) + '_spu_GT_' + str(lbl.squeeze()) + '_T_' + str(self.threshold.squeeze()) + '_input.png',
                         path=samples_out_dir)
                save_img(to_numpy(norm_img).squeeze(), prefix=str(idx) + '_norm_GT_' + str(lbl.squeeze()) + '_T_' + str(self.threshold.squeeze()) + '_input.png',
                         path=samples_out_dir)
                save_img(spu_attr.squeeze(), prefix=str(idx) + '_spu_pred: ' + str(spu_p) + '_mask.png', path=samples_out_dir)
                save_img(norm_attr.squeeze(), prefix=str(idx) + '_norm_pred: ' + str(norm_p) + '_mask.png', path=samples_out_dir)

            if count >= num_samples:
                break
        ncc_values = np.asarray(ncc_values)
        mean_ncc = np.mean(ncc_values)
        if self.attr_method == "attrinet":
            update_key_value_pairs(self.all_results_file, self.model_name, "explanation_ncc", float(mean_ncc))
        else:
            update_key_value_pairs(self.all_results_file, self.model_name, self.attr_method +"_explanation_ncc", float(mean_ncc))
        print("mean_ncc", mean_ncc)

    def get_sensitivity(self, num_samples, positive_only=True):
        sample_idces = self.flip_idx
        count = 0
        sensitivity_values = []

        for i in tqdm(range(len(sample_idces))):
            idx = sample_idces[i]
            spu_data = self.spu_dataloader.dataset[int(idx)]
            spu_img = spu_data['img']
            lbl = spu_data['label'].squeeze()
            spu_img = torch.from_numpy(spu_img[None])

            if self.attr_method == "attrinet":
                spu_y_pred = self.solver.get_probs(spu_img.to(self.device), self.label_idx)
                spu_attr = self.solver.get_attributes(spu_img, self.label_idx)
                spu_attr = -to_numpy(spu_attr).squeeze()

            else:
                spu_y_pred_logits = self.solver.model(spu_img.to(self.device))
                spu_y_pred = torch.sigmoid(spu_y_pred_logits).squeeze()
                spu_attr = self.solver.get_attributes(spu_img, self.label_idx, positive_only=positive_only)
                spu_attr = to_numpy(spu_attr).squeeze()

            if i < 20:
                hitts = self.sstt(spu_attr, confounder=self.confounder, attr_method=self.attr_method, ratio=self.ratio, plot_itsect=True, out_dir=self.out_dir, prefix=str(idx))
            else:
                hitts = self.sstt(spu_attr, confounder=self.confounder, attr_method=self.attr_method, ratio=self.ratio, plot_itsect=False, out_dir=None, prefix=None)
            sensitivity_values.append(hitts)
            count += 1
            if count >= num_samples:
                break

        sensitivity_values = np.asarray(sensitivity_values)
        mean_sensitivity = np.mean(sensitivity_values)
        if self.attr_method == "attrinet":
            update_key_value_pairs(self.all_results_file, self.model_name, "confounder_sensitivity", float(mean_sensitivity))
        else:
            update_key_value_pairs(self.all_results_file, self.model_name, self.attr_method + "_confounder_sensitivity", float(mean_sensitivity))

        print("mean_sensitivity", mean_sensitivity)
    def sstt(self, spu_attr, confounder, attr_method, ratio, plot_itsect, out_dir, prefix):
        #'lime', 'GCam', 'GB', 'shap', 'gifsplanation','attrinet'
        # get confounder pixel positions
        confounder_pos = np.where(confounder == 1)
        confounder_pos_x = confounder_pos[0]
        confounder_pos_y = confounder_pos[1]
        num_founder_pixels = len(confounder_pos_y)
        confounder_pixels = list(zip(confounder_pos_x, confounder_pos_y))
        confounder_set = set(confounder_pixels)

        #select top 10% pixel with highest value
        all_attr_pixel = 320*320
        num_pixels = int(ratio * all_attr_pixel)

        if attr_method == 'attrinet' or attr_method == 'gifsplanation':
            pixel_importance = np.absolute(to_numpy(spu_attr.squeeze()))
        else:
            pixel_importance = spu_attr

        idcs = np.argsort(pixel_importance.flatten())  # from smallest to biggest
        idcs = idcs[::-1]  # if we want the order biggest to smallest, we reverse the indices array
        idcs = idcs[:num_pixels]
        # Compute the corresponding masks for deleting pixels in the given order
        positions = np.array(np.unravel_index(idcs, pixel_importance.shape)).T  # first colum, h index, second column, w index
        attri_pos_x = positions[:, 0]
        attri_pos_y = positions[:, 1]
        top_attri_pixels = list(zip(attri_pos_x, attri_pos_y))
        top_attri_set = set(top_attri_pixels)
        inter_set = confounder_set.intersection(top_attri_set)
        hitts = len(inter_set)/num_founder_pixels
        if hitts!=0 and plot_itsect == True:
            pixels = [list(item) for item in inter_set]
            pixels = np.asarray(pixels)
            background = np.zeros((320, 320))
            background[pixels[:, 0], pixels[:, 1]] = 255
            img = Image.fromarray(background)
            img = img.convert("L")
            out_path = os.path.join(out_dir, prefix + "_" + str(hitts) + "_hitts.png")
            img.save(out_path)
        return hitts







def str2bool(v):
    return v.lower() in ('true')

def argument_parser():
    """
    Create a parser with run_experiments arguments.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="spurious model/sample analyser.")

    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'attrinet'])
    parser.add_argument('--dataset', type=str, default='chexpert', choices=['chexpert', 'vindrcxr'])
    parser.add_argument('--contaminated_class', type=str, default="Cardiomegaly",
                        choices=["Cardiomegaly", "Aortic enlargement"])
    parser.add_argument('--attr_method', type=str, default='GCam',
                        help="choose the explaination methods, can be 'lime', 'GCam', 'GB', 'shap', 'gifsplanation', 'attrinet'")
    parser.add_argument('--positive_only', type=str2bool, default=False,
                        help="if Ture, only select positive attributions, if False, keep all attribution")
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])

    parser.add_argument('--contaim_scale', type=int, default=2, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--contaim_type', type=str, default="tag", choices=["tag", "hyperintensities", "obstruction"])

    parser.add_argument('--num_samples', type=int, default=100, choices=[100],
                        help="number of flipped sample to evaluate on, 100 for short evaluate, 2500 for larger evaluation")

    parser.add_argument("--img_size", default=320,
                        type=int, help="image size for the data loader.")
    parser.add_argument("--batch_size", default=1,
                        type=int, help="Batch size for the data loader.")
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to run on the GPU')
    parser.add_argument('--all_results_file', type=str, default="all_results.json", help='dictionary to save all results')

    return parser


def get_arguments():
    parser = argument_parser()
    exp_configs = parser.parse_args()
    if exp_configs.attr_method == "attrinet":
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
    exp_configs.dataset += '_' + exp_configs.contaminated_class + '_' + exp_configs.contaim_type + '_' + 'degree' + str(
        exp_configs.contaim_scale)  # "chexpert_Cardiomegaly_tag_degree2"

    if "chexpert" in exp_configs.dataset:
        if "Cardiomegaly" in exp_configs.dataset:
            dataset_dict = dataset_dict_chexpert_Cardiomegaly[exp_configs.dataset]
    if "vindrcxr" in exp_configs.dataset:
        dataset_dict = dataset_dict_vindrcxr_Aortic_enlargement[exp_configs.dataset]
    datamodule = prepare_datamodule(exp_configs, dataset_dict)
    exp_configs.train_diseases = dataset_dict["train_diseases"]

    if exp_configs.attr_method == "attrinet":
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
        solver.set_explainer(which_explainer=exp_configs.attr_method)

    exp_configs.result_dir = os.path.join(exp_configs.model_path, "miccai23", "selected_flip_prediction_samples", "new_eval_1615", exp_configs.attr_method)
    os.makedirs(exp_configs.result_dir, exist_ok=True)

    print("exp_configs.result_dir: ",exp_configs.result_dir)
    print("exp_configs.model_path: ",exp_configs.model_path)

    return solver



def prep_dataloaders(exp_configs):

    from data.contaminate_data_settings import TGT_DATA_ROOT
    confounder_dict = {
        "tag": os.path.join(os.path.dirname(os.path.abspath(__file__)), "confounder_masks", "tag.txt"),
        "hyperintensities": os.path.join(os.path.dirname(os.path.abspath(__file__)), "confounder_masks", "hyperintensities.txt"),
        "obstruction": os.path.join(os.path.dirname(os.path.abspath(__file__)), "confounder_masks", "obstruction.txt")
    }

    os.path.join(os.path.dirname(os.path.abspath(__file__)), "confounder_masks", "tag.txt")


    normal_chexpert_Cardiomegaly_root_folders = {
        "tag": os.path.join(TGT_DATA_ROOT, "chexpert", "Cardiomegaly", "tag", "degree0"),
        "hyperintensities": os.path.join(TGT_DATA_ROOT, "chexpert", "Cardiomegaly", "hyperintensities", "degree0"),
        "obstruction": os.path.join(TGT_DATA_ROOT, "chexpert", "Cardiomegaly", "obstruction", "degree0")
    }

    normal_vindr_Aortic_enlargement_root_folders = {
        "tag": os.path.join(TGT_DATA_ROOT, "vindr", "Aortic enlargement", "tag", "degree0"),
        "hyperintensities": os.path.join(TGT_DATA_ROOT, "vindr", "Aortic enlargement", "hyperintensities", "degree0"),
        "obstruction": os.path.join(TGT_DATA_ROOT, "vindr", "Aortic enlargement", "obstruction", "degree0")

    }
    spu_chexpert_Cardiomegaly_root_folders = {
        "tag": os.path.join(TGT_DATA_ROOT, "chexpert", "Cardiomegaly", "tag", "degree4"),
        "hyperintensities": os.path.join(TGT_DATA_ROOT, "chexpert", "Cardiomegaly", "hyperintensities", "degree4"),
        "obstruction": os.path.join(TGT_DATA_ROOT, "chexpert", "Cardiomegaly", "obstruction", "degree4")
    }

    spu_vindr_Aortic_enlargement_root_folders = {
        "tag": os.path.join(TGT_DATA_ROOT, "vindr", "Aortic enlargement", "tag", "degree4"),
        "hyperintensities": os.path.join(TGT_DATA_ROOT, "vindr", "Aortic enlargement", "hyperintensities", "degree4"),
        "obstruction": os.path.join(TGT_DATA_ROOT, "vindr", "Aortic enlargement", "obstruction", "degree4")
    }

    transforms = tfs.Compose([tfs.Resize((exp_configs.img_size, exp_configs.img_size)), tfs.ToTensor()])
    if "chexpert" in exp_configs.dataset:
        normal_data_root = normal_chexpert_Cardiomegaly_root_folders[exp_configs.contaim_type]
        spu_data_root = spu_chexpert_Cardiomegaly_root_folders[exp_configs.contaim_type]

    if "vindrcxr" in exp_configs.dataset:
        normal_data_root = normal_vindr_Aortic_enlargement_root_folders[exp_configs.contaim_type]
        spu_data_root = spu_vindr_Aortic_enlargement_root_folders[exp_configs.contaim_type]

    spu_img_dir = os.path.join(spu_data_root, "test")
    normal_img_dir = os.path.join(normal_data_root, "test")
    spu_csv_path = os.path.join(spu_data_root, "test_df.csv")
    normal_csv_path = os.path.join(normal_data_root, "test_df.csv")
    spu_df = pd.read_csv(spu_csv_path)
    normal_df = pd.read_csv(normal_csv_path)
    if "chexpert" in exp_configs.dataset:
        spu_testset = CheXpert(spu_img_dir, spu_df, exp_configs.train_diseases, transforms=transforms)
        normal_testset = CheXpert(normal_img_dir, normal_df, exp_configs.train_diseases, transforms=transforms)
    if "vindrcxr" in exp_configs.dataset:
        spu_testset = Vindr_CXR(image_dir=spu_img_dir, df=spu_df, train_diseases=exp_configs.train_diseases,
                                transforms=transforms)
        normal_testset = Vindr_CXR(image_dir=normal_img_dir, df=normal_df,
                                   train_diseases=exp_configs.train_diseases,
                                   transforms=transforms)

    spu_test_loader = DataLoader(spu_testset, batch_size=exp_configs.batch_size, shuffle=False)
    normal_test_loader = DataLoader(normal_testset, batch_size=exp_configs.batch_size, shuffle=False)
    confounder = np.loadtxt(confounder_dict[exp_configs.contaim_type])
    return normal_test_loader, spu_test_loader, confounder






def main(config):

    solver = prep_solver(config)
    normal_test_loader, spu_test_loader, confounder = prep_dataloaders(config)
    threshlod_path = os.path.join(config.model_path, "miccai23", "valid", "auc_result_dir", "best_threshold.txt")
    threshold = np.loadtxt(threshlod_path)
    eval_root_dir = os.path.join(config.model_path, "miccai23", "selected_flip_prediction_samples")
    flip_idx = np.loadtxt(os.path.join(eval_root_dir, "flip_idx.txt")).tolist()
    all_neg_idx = np.loadtxt(os.path.join(eval_root_dir, "neg_idx.txt")).tolist()
    all_pos_idx = np.loadtxt(os.path.join(eval_root_dir, "all_pos_idx.txt")).tolist()

    idx_dict = {
        "flip_idx": flip_idx,
        "all_neg_idx": all_neg_idx,
        "all_pos_idx":all_pos_idx
    }

    detector = spurious_detection(solver, spu_test_loader, normal_test_loader, confounder, threshold=threshold,
                                    sample_indices_dict=idx_dict, out_dir=config.result_dir,
                                    attr_method=config.attr_method, config=config)

    detector.get_ncc(positive_only=False, num_samples=config.num_samples)
    detector.get_sensitivity(num_samples=config.num_samples, positive_only=True)


if __name__ == "__main__":
    params = get_arguments()
    main(params)

