from __future__ import division
import torch
import numpy as np
from torch.autograd import Function
from torch.nn.modules.module import Module

class TransOptFunction(Function):
    """
    Class that defines the transport operator layer forward and backward passes
    """
    @staticmethod
    def forward(x, input, Psi):
        """
        Apply the transformation matrix defined by coeff_use and Psi to the
        input latent vector
        """
        assert type(Psi) == torch.tensor
        N = torch.sqrt(torch.tensor([Psi.size()[0]],dtype = torch.int32, device = "cuda"))
        M = torch.tensor([Psi.size], dtype = torch.int, device = "cuda")
        batch_size = torch.tensor([input.size()[0]], device = "cuda")
        ctx.save_for_backward(input, Psi)
        input_pt = torch.tensor([input[:,0:N]], device = "cuda")
        coeff = input[:,N:]
        x1_est = torch.zeros(batch_size[0], N[0], device = "cuda")
        for b_idx in range(0,batch_size[0]):
            x0_use = ##TODO
            coeff_use = ##TODO

            dot_1, indeces = torch.sort(torch.dot(Psi, coeff_use))
            A = torch.reshape(dot_1, (N,N))
            T = torch.real(torch.matrix_exp(A))
            x1_est[b_idx,:] = torch.dot(T, x0_use)[:,0]

        result = x1_est

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the gradient on the transport operator dictionaries
        """
        input, Psi = ctx.saved_tensors
        input_coeff = input
        N = torch.tensor([Psi.size()[0]], device = "cuda")
        M = torch.tensor([Psi.size()[1]], device = "cuda")
        batch_size = torch.tensor([input.size()[0]], device = "cuda")
        input_pt = input_coeff[:, 0:N]
        coeff = input_coeff[:, N:]
        c_grad_total = torch.zeros(batch_size, M, device = "cuda")
        Psi_grad_total = torch.zeros(N*N[0], M, device = "cuda")
        grad_z0_total = torch.zeros(batch_size, N, device = "cuda")


        for b_idx in range(0, batch_size):

            x0_use = ##TODO
            coeff_use = ##TODO
            A, indeces = torch.sort(torch.dot(Psi, coeff_use))
            A = torch.reshape(A, (N,N))
            T = torch.real(torch.matrix_exp(A))
            grad_z1_use = ##TODO

            grad_z0_total[b_idx, :] = torch.dot()##TODO

            eigen = torch.linalg.eig(A)

            U = torch.real(eigen[1])
            D = torch.real(eigen[0])
            V = tprch.transpose(V)
            P = torch.dot(torch.dot(torch.transpose(U), grad_z1_use), torch.dot(torch.transpose(x0_use), grad_z1_use  ))[:, 0]

            F_mat = torch.zeros(D.size()[0], D.size()[0], device = "cuda")

            for a in range (0, D.size()[0]):
                for b in range(0, D.size()[0]):
                    if D[a] == D[b]:
                        F_mat[a, b] = ##TODO
                    else:
                        F_mat[a, b] = ##TODO

            fp = torch.mul(F_mat, P)
            Q1 = torch.dot(V, fp)
            Q = torch.dot(Q1, torch.transpose(U))
            c_grad = torch.dot(torch.reshape(sort(Q), -1), Psi) ##TODO
            c_grad_total[b_idx, :] = torch.real(c_grad)

            Psi_grad_single = torch.zeros(N*N, N, device = "cuda")
            for m in range(0, M):
                Psi_grad_temp = torch.zeros(N,N, device = "cuda")
                for k in range (0, N):
                    for i in range (0, N):
                        Psi_grad_temp[k, i] = torch.real(Q[k, i]*coeff[b_idx, m])
                Psi_grad_single[:, m] = torch.reshape(torch.sort(Psi_grad_temp), (N*N))
            Psi_grad_total = Psi_grad_single + Psi_grad_total

        grad_z_coeff = torch.cat((grad_z0_total, c_grad_total), axis = 1)
        return grad_z_coeff.to(torch.float32), Psi_grad_total.to(torch.float32)


class TransOpt(Module):
    def __init__(self):

        super(TransOpt, self).__init__()

    def forward(self, input_z, coeff, Psi, std):
        """
        Define forward pass of Transport Operator layer

        Input:
            - input_z:  Input latent vector
             - coeff:    Transport operator coefficients defining the transformation matrix
              - Psi:      Current transport operator dictoinary
               - std:      Noise std for posterior sampling (if needed)
        """
        input_z_coeff = torch.cat((input_z, coeff), dim=1)
        z_noNoise = TransOptFunction.apply(input_z_coeff, Psi)
        eps = torch.randn_like(z_noNoise) * std
        z_out = z_noNoise + eps
        return z_out


model = TransOptFunction()

x = model.













