import argparse

def str2bool(v):
    return v.lower() in ('true')

def attrinet_get_parser():
    parser = argparse.ArgumentParser()

    # Debug configuration
    parser.add_argument('--debug', type=str2bool, default=False,
                        help='if true, will automatically set d_iters = 1, set savefrequency=1, easy to run all train step for functional test')

    parser.add_argument('--exp_name', type=str, default='attrinet')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--resume', type=str2bool, default=False, help='Whether to resume training from checkpoint')

    # Data configuration.
    parser.add_argument('--dataset', type=str, default='chexpert', choices=['chexpert', 'vindrcxr'])
    parser.add_argument('--contaminated_class', type=str, default="Cardiomegaly",
                        choices=["Cardiomegaly", "Aortic enlargement"])
    parser.add_argument('--contaim_type', type=str, default='tag', choices=["tag", "hyperintensities", "obstruction"])
    parser.add_argument('--contaim_scale', type=int, default=4, choices=[0,1,2,3,4])
    parser.add_argument('--image_size', type=int, default=320, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')

    # Model configuration.
    # Configurations of latent code generator
    parser.add_argument('--n_fc', type=int, default=8, help='number of fc layer in Intermediate_Generator inside generator')
    parser.add_argument('--n_ones', type=int, default=20, help='number of ones to indicting each task, will affect the latent dim of task vector in generator,default is 20')
    parser.add_argument('--num_out_channels', type=int, default=1, help='number of out channels of generator')

    # Configurations of logistic regression classifier
    parser.add_argument('--lgs_downsample_ratio', type=int, default=32,
                        help='downsampling ratio of logistic regression classifier, can be 4, 8, 16, 32, 64, 80, 160')

    # Configurations of generator
    parser.add_argument('--generator_type', type=str, default='stargan',
                        help='type of the resnet in generator, resnet18, resnet50, stargan, stargan2 ')
    parser.add_argument('--deep_supervise', type=str2bool, default=False, help='if use additional classifier to supervised bottleneck layers')
    parser.add_argument('--G_loss_type', type=str, default='with_center_loss',
                        help='different loss type for generator, without_class_loss, with_class_loss(this one only have classification loss), with_center_loss, with_adv_class_loss(class+adv_calss loss), with_deep_supervision ')

    # Configurations of generator
    parser.add_argument('--lambda_critic', type=float, default=1.0, help='weight for critic loss')
    parser.add_argument('--lambda_1', type=float, default=100, help='weight for l1 loss of disease mask')
    parser.add_argument('--lambda_2', type=float, default=200, help='weight for l1 loss of healthy mask')
    parser.add_argument('--lambda_3', type=float, default=100, help='weight for classification loss')
    parser.add_argument('--lambda_centerloss', type=float, default=0.01, help='weight for center loss of disease mask')

    # Training configuration.
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--cls_iteration', type=int, default=5, help='number of classifier iterations per each generator iter, default=5')
    parser.add_argument('--d_iters', type=int, default=5, help='number of discriminator iterations per each generator iter, default=5')
    parser.add_argument('--num_iters', type=int, default=500000, help='number of total iterations for training G')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--savefrequency', type=int, default=500, help='frequency of saving intermediate results, default = 500')
    parser.add_argument('--validfrequency', type=int, default=1000, help='frequency of validation')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--lgs_lr', type=float, default=0.0001, help='learning rate for logistic regression classifier, previous exp use 0.00025')
    parser.add_argument('--weight_decay_lgs', type=float, default=0.00001, help='weight decay for logistic regression classifier')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam, default 0.9')
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--save_path', type=str, default='',
                        help='path of the exp')

    # Step size.
    parser.add_argument('--sample_step', type=int, default=1000,
                        help='frequency of saving visualization samples, default = 500')
    parser.add_argument('--model_valid_step', type=int, default=1000, help='frequency of validation')
    parser.add_argument('--lr_update_step', type=int, default=1000, help='frequency of learning rate update')

    # Testing configuration.
    parser.add_argument('--test_model_path', type=str, default=None,
                        help='path of the models')

    # Miscellaneous.
    parser.add_argument('--use_wandb', type=str2bool, default=True)
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to run on the GPU')

    return parser




def resnet_get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=str2bool, default=False,
                        help='if true, will automatically set stop_early to 5, easy to run all train step for functional test')

    parser.add_argument('--exp_name', type=str, default='resnet_cls')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--resume', type=str2bool, default=False, help='Whether to resume training from checkpoint')

    # Data configuration.
    parser.add_argument('--dataset', type=str, default='chexpert', choices=['chexpert', 'vindrcxr'])
    parser.add_argument('--contaminated_class', type=str, default="Cardiomegaly",
                        choices=["Cardiomegaly", "Aortic enlargement"])
    parser.add_argument('--contaim_type', type=str, default='hyperintensities', choices=["tag", "hyperintensities", "obstruction"])
    parser.add_argument('--contaim_scale', type=int, default=4, choices=[0,1,2,3,4])

    parser.add_argument('--image_size', type=int, default=320, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size')

    # Training configuration.
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay')
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--save_path', type=str, default='', help='path of the exp')

    # Testing configuration.
    parser.add_argument('--test_model_path', type=str, default='/mnt/qb/work/baumgartner/sun22/TT_interaction_exps/ResNet_rebuttal',
                        help='path of the models')

    # Miscellaneous.
    parser.add_argument('--use_wandb', type=str2bool, default=True)
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to run on the GPU')

    return parser



