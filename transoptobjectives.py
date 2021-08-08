from __future__ import division
import time
import torch
import pykeops
from transopt import TransOpt

from utils import *

def transOptObj_c(c, Psi, x0, x1, zeta):
    """
    Define forward pass for transport operator objective with regularizer on coefficients

    Inputs:
        - c:    Vector of transport operator coefficients [M]
        - Psi:  Transport operator dictionaries [N^2 x M]
        - x0:   Starting point for transport operator path [N]
        - x1:   Ending point for transport operator path [N]
        - zeta: Weight on the l1 co efficient regularizer

    Outputs:
        - objFun: Computed transport operator objective
    """
    N = torch.tensor([torch.sqrt(Psi.size()[0])], device = "cuda")
    coeff_use = ##TODO
    x0_use = ##TODO
    x1_use = ##TODO
    A = torch.reshape(torch.sort(torch.dot(Psi, coeff_use)), (N,N))
    T = torch.real(torch.matrix_exp(A))
    x1_est = torch.dot(T, x0_use)[:, 0]
    objFun = 0.5*torch.linalg.norm(x1 - x1_est)**2 + zeta*torch.sum(torch.abs(c))

    return objFun


def transOptDerv_c(c, Psi, x0, x1, zeta):
    """
    Compute the gradient for the transport operator objective with regularizer on coefficients

    Inputs:
        - c:    Vector of transport operator coefficients [M]
        - Psi:  Transport operator dictionaries [N^2 x M]
        - x0:   Starting point for transport operator path [N]
        - x1:   Ending point for transport operator path [N]
        - zeta: Weight on the l1 coefficient regularizer

    Outputs:
        - c_grad: Gradient of the transport operator objective with repsect to the coefficients
    """
    N = torch.tensor([torch.sqrt(Psi.size()[0])], device="cuda")
    coeff_use =  ##TODO
    x0_use =  ##TODO
    x1_use =  ##TODO
    A = torch.reshape(torch.sort(torch.dot(Psi, coeff_use)), (N, N))
    T = torch.real(torch.matrix_exp(A))

    eig_out = torch.real(torch.linalg.eig(A))
    U = eig_out[1]
    D = eig_out[0]
    V = torch.linalg.inv(U)##TODO
    V = torch.transpose(U)

    innerVal = torch.dot(-x1_use, x0_use) + torch.dot(T, torch.dot(x0_use, torch.transpose(x0_use)))
    P = torch.dot(torch.dot(torch.transpose(T), V))

    F_mat = torch.zeros(D.size()[0], D.size()[0], dtype = torch.complex128, device = "cuda")
    for a in range(0, D.size()[0]):
        for b in range(0, D.size()[0]):
            if D[a] == D[b]:
                F_mat[a, b] = torch.exp(D[a])
            else:
                F_mat[a, b] = (torch.exp(D[b])) - torch.exp(D[a])/(D[b] - D[a])

    fp = torch.mul(F_mat, P)
    Q1 = torch.dot(V, fp)
    Q = torch.dot(Q1, torch.transpose(U))
    c_grad = torch.real(torch.dot(torch.sort(Q), Psi) + zeta*torch.sign(c)##TODO
    return c_grad


def infer_transOpt_coeff(x0, x1, Psi, zeta, randMin, randMax):
    """
    Infer the transport operator coefficients

    Inputs:
        - x0:       Starting point for transport operator path [N]
        - x1:       Ending point for transport operator path [N]
        - Psi:      Transport operator dictionarys [N^2 x M]
        - zeta:     Weight on the l1 coefficient regularizer
        - randMin:  Minimium value for the uniform distribution used to intialize coefficeints
        - randMax:  Maximum value for the uniform distribition used to initializer the coefficeints

    Outputs:
        - c_est:    Final inferred coefficients
        - E:        Final objective function value
        - nit:      Number of inference steps
    """
    M = Psi.size()[1]
    c0 = ##TODO
    opt_out = minimize##TODO
    c_est = ##TODO
    E = ##TODO
    nit = ##TODO
    return c_est, E, nit

def compute_posterior_coeff(z0,z1,Psi_use,post_cInfer_weight,M):
    batch_size = z0.size()[0]
    c_est_mu = torch.zeros(batch_size, M, device = "cuda")
    E_mu = torch.zeros(batch_size, 1, device = "cuda")
    nit_mu = torrch.zeros(batch_size, 1, device = "cuda")
    c_infer_time_post = torch.zeros(batch_size, 1, device = "cuda")

    for b in range(0,batch_size):
        c_infer_time_start = time.time()
        x0 = torch.tensor([z0[b,:]], dtype = torch.double, device = "cuda")
        x1 = torch.tensor([z1[b,:]], dtype = torch.double, device = "cuda")
        c_est_mu[b, :], E_mu[b], nit_mu[b] = ##TODO
        c_infer_time_post[b] = time.time() - c_infer_time_start

    return c_est_mu, E_mu, nit_mu, c_infer_time_post

def compute_prior_obj(z_scale,Psi,a_mu_scale,sample_labels_batch,transNet,scale,prior_l1_weight,prior_weight,opt):
    # Initialize arrays for saving inference details
    prior_to_sum = 0.0
    c_est_batch = torch.zeros(opt.batch_size, opt.num_anchor, opt.M, device = "cuda")
    E_anchor = torch.zeros(opt.batch_size, opt.num_anchor, opt.M, device = "cuda")
    nit_anchor = torch.zeros(opt.batch_size, opt.num_anchor, opt.numRestart, device = "cuda")
    c_est_a_store = torch.zeros(opt.batch_size,opt.num_anchor,opt.numRestart,opt.M, device = "cuda")
    anchor_idx_use = torch.zeros(opt.batch_size, device = "cuda")
    for b in range(0, opt.batch_size):
        x1 = z1[b, :].to(torch.double)
        prior_to_anchor_sum = 0.0
        c_est_a = torch.zeros(opt.num_anchor, opt.M, device = "cuda")

        # Specify the anchors that are compared to each sample# Specify the anchors that are compared to each sample
        if opt.


