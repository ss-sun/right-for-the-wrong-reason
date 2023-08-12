from solvers.resnet_solver import resnet_solver
from solvers.attrinet_solver import task_switch_solver
import logging
from experiment_utils import init_seed, init_experiment, init_wandb
from train_utils import prepare_datamodule
import os

EXPS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

def prepare_exps(exp_configs):
    if exp_configs.mode == 'train':
        logging.info("training model: ")
        init_experiment(exp_configs)
    init_seed(exp_configs.manual_seed)
    if exp_configs.use_wandb:
        init_wandb(exp_configs)



def main(exp_configs):

    from data.dataset_params import dataset_dict_chexpert_Cardiomegaly, dataset_dict_vindrcxr_Aortic_enlargement
    exp_configs.dataset += '_'+ exp_configs.contaminated_class + '_'+ exp_configs.contaim_type + '_' + 'degree'+str(exp_configs.contaim_scale)

    prepare_exps(exp_configs)
    logging.info("result folder: " + exp_configs.exp_dir)
    if "chexpert" in exp_configs.dataset:
        if "Cardiomegaly" in exp_configs.dataset:
            dataset_dict = dataset_dict_chexpert_Cardiomegaly[exp_configs.dataset]
    if "vindrcxr" in exp_configs.dataset:
        dataset_dict = dataset_dict_vindrcxr_Aortic_enlargement[exp_configs.dataset]
    datamodule = prepare_datamodule(exp_configs, dataset_dict)
    print(exp_configs)

    data_loader = {}
    if "resnet" in exp_configs.exp_name:
        logging.info("working on resnet")

        data_loader["train"] = datamodule.train_dataloader(batch_size=exp_configs.batch_size, shuffle=True)
        data_loader["valid"] = datamodule.valid_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        data_loader["test"] = datamodule.test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        solver = resnet_solver(exp_configs, data_loader=data_loader)

    if "attrinet" in exp_configs.exp_name:
        logging.info("working on attrinet")

        train_loaders = datamodule.single_disease_train_dataloaders(batch_size=exp_configs.batch_size, shuffle=True)
        vis_dataloaders = datamodule.single_disease_vis_dataloaders(batch_size=4, shuffle=False)
        val_loader = datamodule.valid_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        test_loader = datamodule.test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)

        data_loader['train_pos'] = train_loaders['pos']
        data_loader['train_neg'] = train_loaders['neg']
        data_loader['vis_pos'] = vis_dataloaders['pos']
        data_loader['vis_neg'] = vis_dataloaders['neg']
        data_loader['valid'] = val_loader
        data_loader['test'] = test_loader
        solver = task_switch_solver(exp_configs, data_loader=data_loader)

    if exp_configs.mode == "train":
        print('start training...')
        solver.train()
        print('finish training!')

    if exp_configs.mode == 'test':
        print('start testing....')
        solver.load_model(exp_configs.test_model_path)
        test_auc = solver.test()
        print('finish test!')
        print('test_auc: ', test_auc)


if __name__ == '__main__':
    which_model = 'attrinet'
    if which_model == 'resnet':
        from parser import resnet_get_parser
        parser = resnet_get_parser()
    if which_model == 'attrinet':
        from parser import attrinet_get_parser
        parser = attrinet_get_parser()
    config = parser.parse_args()
    config.save_path = os.path.join(EXPS_ROOT, which_model+"_exps")
    main(config)