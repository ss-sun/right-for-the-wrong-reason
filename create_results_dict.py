import json
import os

eval_models = ['resnet', "attrinet"]
eval_datasets = ['chexpert_Cardiomegaly']
eval_confounders = ['tag', 'hyperintensities', 'obstruction']
eval_contaim_scales = [0, 1, 2, 3, 4]
cls_related = ['valid_auc', 'test_auc', 'threshold']
explainers = ['GB', 'GCam', 'lime', 'shap', 'gifsplanation']
eval_metrics = ['confounder_sensitivity', 'explanation_ncc']

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_results.json")

def create_result_dict(file_path):
    result_dict = {}
    for model in eval_models:
        for dataset in eval_datasets:
            for confounder in eval_confounders:
                for scale in eval_contaim_scales:
                    model_key = model + "_" + dataset + "_" + confounder + "_" + "degree"+str(scale)
                    model_result_dict = {}
                    for cls in cls_related:
                        model_result_dict[cls] = float('nan')
                    if model == 'resnet':
                        for explainer in explainers:
                                for metric in eval_metrics:
                                    model_result_dict[explainer + "_" + metric] = float('nan')
                    if model == 'attrinet':
                        for metric in eval_metrics:
                            model_result_dict[metric] = float('nan')
                    result_dict[model_key] = model_result_dict
    with open(file_path, "w") as file:
        json.dump(result_dict, file, indent=4)



def add_key_value_pairs(filename, new_data):
    with open(filename, "r") as file:
        data = json.load(file)
    data.update(new_data)
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def update_key_value_pairs(filename, model_name, measure_name, value):
    with open(filename, "r") as file:
        data = json.load(file)
    dict = data[model_name]
    dict[measure_name] = value
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

create_result_dict(file_path)
# update_key_value_pairs(file_path, model_name="resnet_chexpert_Pneumothorax_stripe_degree0.0", measure_name="valid_auc", value=float('nan'))