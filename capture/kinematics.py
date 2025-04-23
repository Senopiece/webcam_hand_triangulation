import torch

from emg2pose.kinematics import forward_kinematics, load_default_hand_model
from .hand_normalization import normalize_hands

hand_model = load_default_hand_model()

POINTS_SELECT = [5, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4]


# x: B, C
def forward_hand_kinematics(x: torch.Tensor):
    hands = forward_kinematics(x.unsqueeze(1).permute(0, 2, 1), hand_model).squeeze(1)
    return hands[:, POINTS_SELECT, :]  # B, L, 3


def inverse_hand_kinematics(y: torch.Tensor, x_init: None | torch.Tensor = None):
    # y: B, L, 3

    scaler = 90.0

    with torch.inference_mode():
        y = scaler * normalize_hands(y)

    B, L = y.shape[0], y.shape[1]
    C = 20

    if x_init is not None:
        x = x_init.clone().detach().to(y.device).requires_grad_(True)
    else:
        x = torch.zeros(B, C, device=y.device, requires_grad=True)

    # lr estimated from initial error
    y_hat = scaler * normalize_hands(forward_hand_kinematics(x))
    mx_s_err = ((y_hat - y) ** 2).sum(dim=-1).max()
    mx_err = mx_s_err.sqrt()
    mx_s_err = mx_s_err.item()

    optimizer = torch.optim.NAdam([x], lr=0.6)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=12,
    )

    for _ in range(12):
        optimizer.zero_grad()
        y_hat = scaler * normalize_hands(forward_hand_kinematics(x))
        loss = ((y_hat - y) ** 2).sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        mx_err = ((y_hat - y) ** 2).sum(dim=-1).max().sqrt().item()

        if mx_err < 1:
            break

    return x  # x: B, C
