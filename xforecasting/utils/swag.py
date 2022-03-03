#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 17:48:21 2022

@author: ghiggi
"""
# SWAG Implementation
import torch
import numpy as np
from ..dataloader_autoregressive import get_aligned_ar_batch
from ..dataloader_autoregressive import AutoregressiveDataset
from ..dataloader_autoregressive import AutoregressiveDataLoader

# import itertools
# from torch.distributions.normal import Normal
# import gpytorch
# from gpytorch.lazy import RootLazyTensor, DiagLazyTensor, AddedDiagLazyTensor
# from gpytorch.distributions import MultivariateNormal

## https://pytorch.org/docs/master/optim.html#stochastic-weight-averaging
# TOCHECK:
# check_bn : can found also BN enclosed in ResBlock --> ConvBlock (nestdness?)


def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList


def swag_parameters(module, params, no_cov_mat=True):
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            continue
        data = module._parameters[name].data
        module._parameters.pop(name)
        module.register_buffer("%s_mean" % name, data.new(data.size()).zero_())
        module.register_buffer("%s_sq_mean" % name, data.new(data.size()).zero_())

        if no_cov_mat is False:
            module.register_buffer(
                "%s_cov_mat_sqrt" % name, data.new_empty((0, data.numel())).zero_()
            )

        params.append((module, name))


class SWAG(torch.nn.Module):
    def __init__(
        self, base, no_cov_mat=True, max_num_models=0, var_clamp=1e-30, *args, **kwargs
    ):
        super(SWAG, self).__init__()

        self.register_buffer("n_models", torch.zeros([1], dtype=torch.long))
        self.params = list()

        self.no_cov_mat = no_cov_mat
        self.max_num_models = max_num_models

        self.var_clamp = var_clamp

        self.base = base(*args, **kwargs)
        self.base.apply(
            lambda module: swag_parameters(
                module=module, params=self.params, no_cov_mat=self.no_cov_mat
            )
        )

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def sample(self, scale=1.0, cov=False, seed=None, block=False, fullrank=True):
        if seed is not None:
            torch.manual_seed(seed)

        if not block:
            self.sample_fullrank(scale, cov, fullrank)
        else:
            self.sample_blockwise(scale, cov, fullrank)

    def sample_blockwise(self, scale, cov, fullrank):
        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)

            sq_mean = module.__getattr__("%s_sq_mean" % name)
            eps = torch.randn_like(mean)

            var = torch.clamp(sq_mean - mean**2, self.var_clamp)

            scaled_diag_sample = scale * torch.sqrt(var) * eps

            if cov is True:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                eps = cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0), 1)).normal_()
                cov_sample = (
                    scale / ((self.max_num_models - 1) ** 0.5)
                ) * cov_mat_sqrt.t().matmul(eps).view_as(mean)

                if fullrank:
                    w = mean + scaled_diag_sample + cov_sample
                else:
                    w = mean + scaled_diag_sample

            else:
                w = mean + scaled_diag_sample

            module.__setattr__(name, w)

    def sample_fullrank(self, scale, cov, fullrank):
        scale_sqrt = scale**0.5

        mean_list = []
        sq_mean_list = []

        if cov:
            cov_mat_sqrt_list = []

        for (module, name) in self.params:
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            if cov:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                cov_mat_sqrt_list.append(cov_mat_sqrt.cpu())

            mean_list.append(mean.cpu())
            sq_mean_list.append(sq_mean.cpu())

        mean = flatten(mean_list)
        sq_mean = flatten(sq_mean_list)

        # draw diagonal variance sample
        var = torch.clamp(sq_mean - mean**2, self.var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        # if covariance draw low rank sample
        if cov:
            cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)

            cov_sample = cov_mat_sqrt.t().matmul(
                cov_mat_sqrt.new_empty(
                    (cov_mat_sqrt.size(0),), requires_grad=False
                ).normal_()
            )
            cov_sample /= (self.max_num_models - 1) ** 0.5

            rand_sample = var_sample + cov_sample
        else:
            rand_sample = var_sample

        # update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # unflatten new sample like the mean sample
        samples_list = unflatten_like(sample, mean_list)

        for (module, name), sample in zip(self.params, samples_list):
            module.__setattr__(name, sample.cuda())

    def collect_model(self, base_model):
        for (module, name), base_param in zip(self.params, base_model.parameters()):
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            # first moment
            mean = mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + base_param.data / (self.n_models.item() + 1.0)

            # second moment
            sq_mean = sq_mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + base_param.data**2 / (self.n_models.item() + 1.0)

            # square root of covariance matrix
            if self.no_cov_mat is False:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)

                # block covariance matrices, store deviation from current mean
                dev = (base_param.data - mean).view(-1, 1)
                cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1, 1).t()), dim=0)

                # remove first column if we have stored too many models
                if (self.n_models.item() + 1) > self.max_num_models:
                    cov_mat_sqrt = cov_mat_sqrt[1:, :]
                module.__setattr__("%s_cov_mat_sqrt" % name, cov_mat_sqrt)

            module.__setattr__("%s_mean" % name, mean)
            module.__setattr__("%s_sq_mean" % name, sq_mean)
        self.n_models.add_(1)

    def load_state_dict(self, state_dict, strict=True):
        if not self.no_cov_mat:
            n_models = state_dict["n_models"].item()
            rank = min(n_models, self.max_num_models)
            for module, name in self.params:
                mean = module.__getattr__("%s_mean" % name)
                module.__setattr__(
                    "%s_cov_mat_sqrt" % name,
                    mean.new_empty((rank, mean.numel())).zero_(),
                )
        super(SWAG, self).load_state_dict(state_dict, strict)

    def export_numpy_params(self, export_cov_mat=False):
        mean_list = []
        sq_mean_list = []
        cov_mat_list = []

        for module, name in self.params:
            mean_list.append(module.__getattr__("%s_mean" % name).cpu().numpy().ravel())
            sq_mean_list.append(
                module.__getattr__("%s_sq_mean" % name).cpu().numpy().ravel()
            )
            if export_cov_mat:
                cov_mat_list.append(
                    module.__getattr__("%s_cov_mat_sqrt" % name).cpu().numpy().ravel()
                )
        mean = np.concatenate(mean_list)
        sq_mean = np.concatenate(sq_mean_list)
        var = sq_mean - np.square(mean)

        if export_cov_mat:
            return mean, var, cov_mat_list
        else:
            return mean, var

    def import_numpy_weights(self, w):
        k = 0
        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)
            s = np.prod(mean.shape)
            module.__setattr__(name, mean.new_tensor(w[k : k + s].reshape(mean.shape)))
            k += s

    def generate_mean_var_covar(self):
        mean_list = []
        var_list = []
        cov_mat_root_list = []
        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)
            cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)

            mean_list.append(mean)
            var_list.append(sq_mean - mean**2.0)
            cov_mat_root_list.append(cov_mat_sqrt)
        return mean_list, var_list, cov_mat_root_list


###---------------------------------------------------------------------------.
###########################
### Batch Norm Updates ####
###########################
def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(
    model,
    # Data
    data_dynamic,
    data_static=None,
    data_bc=None,
    bc_generator=None,
    # AR_batching_function
    ar_batch_fun=get_aligned_ar_batch,
    # Scaler options
    scaler=None,
    # Dataloader options
    batch_size=64,
    num_workers=0,
    prefetch_factor=2,
    prefetch_in_gpu=False,
    pin_memory=False,
    asyncronous_gpu_transfer=True,
    device="cpu",
    numeric_precision="float32",
    # Autoregressive settings
    input_k=[-3, -2, -1],
    output_k=[0],
    forecast_cycle=1,
    ar_iterations=2,
    stack_most_recent_prediction=True,
    **kwargs
):
    """
    BatchNorm buffers update (if any).
    Performs 1 epochs to estimate buffers average using train dataset.
    :param model: model being update
    :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))

    dataset = AutoregressiveDataset(
        data_dynamic=data_dynamic,
        data_bc=data_bc,
        data_static=data_static,
        bc_generator=bc_generator,
        scaler=scaler,
        # Custom AR batching function
        ar_batch_fun=ar_batch_fun,
        # Autoregressive settings
        input_k=input_k,
        output_k=output_k,
        forecast_cycle=forecast_cycle,
        ar_iterations=ar_iterations,
        stack_most_recent_prediction=stack_most_recent_prediction,
        # GPU settings
        training_mode=False,
        device=device,
    )

    dataloader = AutoregressiveDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last_batch=False,
        random_shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        prefetch_in_gpu=prefetch_in_gpu,
        pin_memory=pin_memory,
        asyncronous_gpu_transfer=asyncronous_gpu_transfer,
        device=device,
    )

    n = 0
    ##------------------------------------------------------------------------.
    # Retrieve custom ar_batch_fun fuction
    ar_batch_fun = dataset.ar_batch_fun
    with torch.no_grad():
        ##--------------------------------------------------------------------.
        # Iterate along training batches
        for batch_dict in dataloader:
            # batch_dict = next(iter(batch_dict))
            ##----------------------------------------------------------------.
            ### Perform autoregressive loop
            dict_Y_predicted = {}
            for ar_iteration in range(ar_iterations + 1):
                # Retrieve X and Y for current AR iteration
                # - Torch Y stays in CPU with training_mode=False
                torch_X, _ = ar_batch_fun(
                    ar_iteration=ar_iteration,
                    batch_dict=batch_dict,
                    dict_Y_predicted=dict_Y_predicted,
                    device=device,
                    asyncronous_gpu_transfer=asyncronous_gpu_transfer,
                )

                input_var = torch.autograd.Variable(torch_X)
                b = input_var.data.size(0)

                momentum = b / (n + b)
                for module in momenta.keys():
                    module.momentum = momentum

                dict_Y_predicted[ar_iteration] = model(input_var, **kwargs)
                n += b
                del torch_X, input_var

    model.apply(lambda module: _set_momenta(module, momenta))


def bn_update_with_loader(
    model,
    loader,
    ar_iterations=2,
    asyncronous_gpu_transfer=True,
    device="cpu",
    **kwargs
):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    # Retrieve custom ar_batch_fun fuction
    ar_batch_fun = loader.ar_batch_fun
    with torch.no_grad():
        ##--------------------------------------------------------------------.
        # Iterate along training batches
        for batch_dict in loader:
            # batch_dict = next(iter(batch_dict))
            ##----------------------------------------------------------------.
            ### Perform autoregressive loop
            dict_Y_predicted = {}
            for ar_iteration in range(ar_iterations + 1):
                # Retrieve X and Y for current AR iteration
                # - Torch Y stays in CPU with training_mode=False
                torch_X, _ = ar_batch_fun(
                    ar_iteration=ar_iteration,
                    batch_dict=batch_dict,
                    dict_Y_predicted=dict_Y_predicted,
                    device=device,
                    asyncronous_gpu_transfer=asyncronous_gpu_transfer,
                )

                input_var = torch.autograd.Variable(torch_X)
                b = input_var.data.size(0)

                momentum = b / (n + b)
                for module in momenta.keys():
                    module.momentum = momentum

                dict_Y_predicted[ar_iteration] = model(input_var, **kwargs)
                n += b
                del torch_X, input_var

            del dict_Y_predicted

    model.apply(lambda module: _set_momenta(module, momenta))
