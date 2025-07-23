'''
config:
    * num_queries:          int
    * camera_names:         list[str]
    * vq:                   bool (False)
    * vq_class:             int (0)
    * vq_dim:               int (0)
    * action_dim:           int (16)
    * lr_backbone:          float (1e-5)
    * backbone:             str ('resnet18')
    * dilation:             bool (False)
    * masks:                bool (False)
    * hidden_dim:           int (512)
    * position_embedding:   str ('sine')
    * dropout:              float (0.1)
    * nheads:               int (8)
    * dim_feedforward:      int (3200)
    * enc_layers:           int (4)
    * dec_layers:           int (7)
    * pre_norm:             bool (False)
    * no_encoder:           bool (False)
'''
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
from .build import build_ACT_model_and_optimizer, build_ACT_model


class ACTPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(config)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = config['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, vq_sample=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            loss_dict = dict()
            a_hat, _, (mu, logvar), _, _ = self.model(qpos, image, env_state, actions, is_pad, vq_sample)
            total_kld, _, _ = kl_divergence(mu, logvar)
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _), _, _ = self.model(qpos, image, env_state, vq_sample=vq_sample) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

    @torch.no_grad()
    def vq_encode(self, qpos, actions, is_pad):
        actions = actions[:, :self.model.num_queries]
        is_pad = is_pad[:, :self.model.num_queries]

        _, _, binaries, _, _ = self.model.encode(qpos, actions, is_pad)

        return binaries
        
    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)
    

class ControllerACTPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = build_ACT_model(config)
    
    def __call__(self, qpos, image):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        a_hat, _, (_, _), _, _ = self.model(qpos, image, None) # no action, sample from prior
        return a_hat

    def serialize(self):
        return self.state_dict()
    
    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
