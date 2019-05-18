import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from u_net import u_net
from utils.data_helper import load_data, get_input
from utils.audio_helper import mix_specs, get_mask, spec_normalization, inv_sample, spec_to_frag

SPEC_DIR =  '../dataset/trainset/spectrum'
MODEL_DIR = '../dataset/model'

def step_train(specs, labels, mask_net, criterion, mask_optimizer, device, N=2, validate=False):
    # initialize return values
    loss = 0.

    # mix audio and get ground truth mask
    specs_mixed = mix_specs(specs)
    masks_gt = torch.from_numpy(get_mask(specs)).to(device)
    masks_predicted = torch.zeros(masks_gt.shape).to(device)

    # get input
    audio_input = torch.from_numpy(spec_normalization(specs_mixed)).to(device)

    # get output
    audio_output = mask_net.forward(audio_input)

    # extract predicted masks
    for batch in range(labels.shape[0]):
        for n in range(labels.shape[1]):
            masks_predicted[batch, n, :, :] = audio_output[batch, labels[batch, n], :, :]
    
    # get loss
    loss = criterion(masks_predicted, masks_gt)

    # get gradient and update
    mask_net.zero_grad()
    loss.backward()
    mask_optimizer.step()

    # return
    if validate == False:
        return loss.detach().cpu().numpy()
    else:
        np.save('./validate/masks_gt', masks_gt.detach().cpu().numpy())
        np.save('./validate/masks_predicted', masks_predicted.detach().cpu().numpy())
        masks_inv_sampled = inv_sample(np.rint(masks_predicted.detach().cpu().numpy()))
        specs_predicted = specs_mixed * masks_inv_sampled
        return [loss.detach().cpu().numpy(), specs_predicted, specs_mixed]

def train(spec_dir, model_dir=None, epoch_num=10, validate_freq=200):
    # define networks
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    mask_net = u_net(16, 8).to(device)

    # define criterion
    criterion = nn.BCELoss()

    # define optimizers
    mask_optimizer = optim.SGD(mask_net.parameters(), lr=0.001, momentum=0.9)

    # load parameters
    if os.path.exists(os.path.join(model_dir, 'mask_net_param.pkl')):
        mask_net.load_state_dict(torch.load(os.path.join(model_dir, 'mask_net_param.pkl')))
        mask_net.eval()
    
    # load training data
    print('data loading...')
    specs = load_data(spec_dir)
    print('data loaded!')

    # begin training
    for epoch in range(epoch_num):
        loss = 0.
        for step in range(2000):
            # sample input
            specs_input, labels_input = get_input(specs, batch_size)

            # if not validate
            if step % validate_freq != 0:
                if not (specs_input is None or labels_input is None):
                    # train one step
                    loss += step_train(specs_input, labels_input, mask_net, criterion, mask_optimizer, device)
            
            # validate
            else:
                if not (specs_input is None or labels_input is None):
                    # train one step
                    step_loss, specs_predicted, specs_mixed = step_train(specs_input, labels_input, mask_net, criterion, mask_optimizer, device, validate=True)
                    loss += step_loss

                    # evaluate performance
                    print('epoch: %2d   step: %4d   loss: %f' % (epoch, step, loss/(step+1)))            
        
        if os.path.exists(model_dir) == False:
            os.mkdir(model_dir)
        torch.save(mask_net.state_dict(), os.path.join(model_dir, 'mask_net_param.pkl'))
        if epoch % 10 == 9:
            torch.save(mask_net.state_dict(), os.path.join(model_dir, 'mask_net_param_' + str(epoch) + '.pkl'))

if __name__ == '__main__':
    train(SPEC_DIR, MODEL_DIR, epoch_num=100, validate_freq=200)
