import os
from copy import deepcopy as copy
from data.contaminate_data_settings import TGT_DATA_ROOT

def update_default(params):
    default_config = {}
    exp = copy(default_config)
    exp.update(params)
    return exp


chexpert_datasets_names = []
for dataset_name in ["chexpert"]:
    for disease in ["Cardiomegaly"]:
        for signal in ["tag", "hyperintensities", "obstruction"]:
            for degree in [0,1,2,3,4]:
                chexpert_datasets_names.append("{dataset_name}_{disease}_{signal}_degree{degree}".format(dataset_name=dataset_name, disease=disease, signal=signal, degree=degree))


vindr_datasets_names = []
for dataset_name in ["vindrcxr"]:
    for disease in ["Aortic enlargement"]:
        for signal in ["tag", "hyperintensities", "obstruction"]:
            for degree in [0,1,2,3,4]:
                vindr_datasets_names.append("{dataset_name}_{disease}_{signal}_degree{degree}".format(dataset_name=dataset_name, disease=disease, signal=signal, degree=degree))



dataset_dict_chexpert_Cardiomegaly = {
    "{name}".format(name=name): update_default({
        "dataset_dir": os.path.join(TGT_DATA_ROOT, name.split("_")[0], name.split("_")[1], name.split("_")[2], name.split("_")[3]),
        "train_diseases": ["Cardiomegaly"],
        "img_size": 320,
    }) for name in chexpert_datasets_names
}

dataset_dict_vindrcxr_Aortic_enlargement = {
    "{name}".format(name=name): update_default({
        "dataset_dir": os.path.join(TGT_DATA_ROOT, name.split("_")[0], name.split("_")[1], name.split("_")[2], name.split("_")[3]),
        "train_diseases": ["Aortic enlargement"],
        "img_size": 320,
    }) for name in vindr_datasets_names
}


