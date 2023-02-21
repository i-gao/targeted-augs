import torch
import IPython
# Missing 90deg rotations and horizontal/vertical flips (used in the orig paper)
# Missing separate brightness and contrast perturbations (used in the orig paper)

class StainColorJitter(object):
    # This is the normalized OD matrix
    # using notation in appendix B of https://ieeexplore.ieee.org/document/8327641
    # and taking the matrix from
    # https://www.researchgate.net/publication/319879820_Quantification_of_histochemical_staining_by_color_deconvolution
    M = torch.tensor([
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78]])
    Minv = torch.inverse(M)
    eps = 1e-6

    def __init__(self, sigma=0.05):
        # Sigma specifies the strength of the augmentation
        self.sigma = sigma

    def __call__(self, P):
        # Expects P to be the result of ToTensor, i.e., each pixel in [0, 1]
        # IPython.embed()
        assert P.shape == (3, 96, 96)
        assert torch.max(P) <= 1.0
        assert torch.min(P) >= 0.0

        # Eqn 5
        S = - (torch.log(255 * P.T + self.eps)).matmul(self.Minv) # 96 x 96 x 3

        # alpha is uniform from U(1 - sigma, 1 + sigma)
        alpha = 1 + (torch.rand(3) - 0.5) * 2 * self.sigma

        # beta is uniform from U(-sigma, sigma)
        beta = (torch.rand(3) - 0.5) * 2 * self.sigma

        # Eqn 6
        Sp = S * alpha + beta

        # Eqn 7
        Pp = torch.exp(-Sp.matmul(self.M)) - self.eps

        # Transpose, rescale, and clip
        Pp = Pp.T / 255
        Pp = torch.clip(Pp, 0.0, 1.0)

        return Pp
