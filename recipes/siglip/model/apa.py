import torch
import torch.nn as nn


class UniRect_static(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lambda_param, kappa_param):
        # normal forward pass
        input, lambda_param, kappa_param = input.detach(), lambda_param.detach(), kappa_param.detach()
        p = torch.exp(
            (1 / lambda_param)
            * torch.nn.functional.softplus(kappa_param * input - torch.log(lambda_param), beta=-1.0, threshold=20)
        )
        ctx.save_for_backward(input, lambda_param, kappa_param)
        out = p * input

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, lambda_param, kappa_param = ctx.saved_tensors
        p = torch.exp(
            (1 / lambda_param)
            * torch.nn.functional.softplus(kappa_param * input - torch.log(lambda_param), beta=-1.0, threshold=20)
        )
        sigmoidal_coeff = 1 / (lambda_param + torch.exp(kappa_param * input))

        part_grad_kappa = (input**2) * sigmoidal_coeff * p
        part_grad_lambda = (-input * p / lambda_param) * (torch.log(p) + sigmoidal_coeff)
        part_grad_input = p * (kappa_param * input * sigmoidal_coeff + 1)

        grad_input = grad_output * part_grad_input
        grad_lambda = grad_output * part_grad_lambda
        grad_kappa = grad_output * part_grad_kappa

        return grad_input, grad_lambda, grad_kappa


class UniRect(nn.Module):
    __constants__ = ["num_parameters"]
    num_parameters: int

    def __init__(
        self,
        num_parameters: int = 1,
        lambda_init=(0.8, 1.2),
        kappa_init=(0.8, 1.2),
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_parameters = num_parameters
        super().__init__()
        lambda_param = torch.nn.init.uniform_(
            torch.empty(num_parameters, **factory_kwargs), a=lambda_init[0], b=lambda_init[1]
        )
        kappa_param = torch.nn.init.uniform_(
            torch.empty(num_parameters, **factory_kwargs), a=kappa_init[0], b=kappa_init[1]
        )
        if num_parameters > 1:
            self.lambda_param = nn.Parameter(lambda_param.unsqueeze(0).unsqueeze(0))
            self.kappa_param = nn.Parameter(kappa_param.unsqueeze(0).unsqueeze(0))
        else:
            self.lambda_param = nn.Parameter(lambda_param)
            self.kappa_param = nn.Parameter(kappa_param)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        lambda_param = torch.clamp(self.lambda_param, min=1e-7)
        out = UniRect_static.apply(input, lambda_param, self.kappa_param)

        return out

    def extra_repr(self) -> str:
        return "num_parameters={}".format(self.num_parameters)
