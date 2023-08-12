import wandb
import matplotlib.pyplot as plt
import torch
import numpy as np
from data.chexpert_data_module import CheXpertDataModule
from data.vindrcxr_data_module import Vindr_CXRDataModule

def prepare_datamodule(exp_configs, dataset_dict):
    # prepare dataloaders
    exp_configs.train_diseases = dataset_dict["train_diseases"]
    if 'chexpert' in exp_configs.dataset:
        print("working on chexpert dataset")
        datamodule = CheXpertDataModule(dataset_dict,
                                        img_size=dataset_dict['img_size'],
                                        seed=exp_configs.manual_seed)

    if 'vindrcxr' in exp_configs.dataset:
        print("working on vindrcxr dataset")
        datamodule = Vindr_CXRDataModule(dataset_dict,
                                        img_size=dataset_dict['img_size'],
                                        seed=exp_configs.manual_seed)
    datamodule.setup()
    return datamodule


def logscalar(name, value):
    wandb.log({name: value})


def print_network(model):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))



def to_numpy(tensor):
    """
    Converting tensor to numpy.
    Args:
        tensor: torch.Tensor
    Returns:
        Tensor converted to numpy.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().cpu().numpy()



def save_batch(img_batch, label_batch, pred_batch=None, out_dir=''):

    vmax = np.abs(img_batch).flatten().max()
    vmin = np.abs(img_batch).flatten().min()
    cols = int(img_batch.shape[0] / 2)
    rows = 2
    figure = plt.figure(figsize=(5*cols, 3*rows))
    for i in range(1, cols * rows + 1):
        img = img_batch[i-1]
        label = label_batch[i-1]
        figure.add_subplot(rows, cols, i)
        title = 'label: ' + str(label)
        if pred_batch is not None:
            pred = pred_batch[i-1]
            title = title + '  pred: '+ str(pred.squeeze())[:4]
        plt.title(title)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)

    plt.savefig(out_dir, bbox_inches='tight')


def save_masks(input_batch, mask_batch, dest_batch, label_batch, pred_batch, num_masks, disease_class, out_dir, scale_intensity=True):

    ## we only save part of the batch, here 4 samples
    #rows = 4
    rows = input_batch.size(0)
    ## for each row, input image, mask1, mask2,..., dest image.
    cols = num_masks + 2
    figure = plt.figure(figsize=(3*cols, 3*rows))

    i = 1
    for sample_idx in range(0, rows):
        input = input_batch[sample_idx]
        label = label_batch[sample_idx]
        pred = pred_batch[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label.numpy())
        plt.axis("off")
        plt.imshow(input.squeeze(), cmap="gray")
        # plt.imshow(input.squeeze(), cmap="gray",vmin=0, vmax=255)
        i = i + 1
        mask = mask_batch[sample_idx]
        for j in np.arange(num_masks):
            img = mask[j]
            if scale_intensity:
                '''
                output masks have value between -2 to 2, therefore here devide 4 and plus 0.5 to change 
                the value range to (0,1). then change to range (0,255)
                '''
                img = -img / 4.0 + 0.5
                img = (img * 255).type(torch.int)
            figure.add_subplot(rows, cols, i)
            plt.title(disease_class[j][:4] + ': ' + str(int(label[j].numpy())) + ', pred: ' + str(float(pred[j].numpy()))[:4])
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=255)
            i = i + 1

        dest = dest_batch[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title('dest')
        plt.axis("off")
        plt.imshow(dest.squeeze(), cmap="gray")
        i = i + 1


    # plt.savefig('test.png')
    plt.savefig(out_dir)













