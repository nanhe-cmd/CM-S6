import torch


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


def selective_scan_pytorch(x, delta, a, b, c, d):
    batch, length, hidden = x.shape
    state_dim = a.shape[1]

    delta_a = torch.exp(delta.unsqueeze(-1) * a.unsqueeze(0).unsqueeze(0))
    delta_b = delta.unsqueeze(-1) * b.unsqueeze(2)
    input_term = x.unsqueeze(-1) * delta_b

    state = torch.zeros(batch, hidden, state_dim, device=x.device, dtype=x.dtype)
    outputs = []
    for index in range(length):
        state = delta_a[:, index] * state + input_term[:, index]
        outputs.append((state * c[:, index].unsqueeze(1)).sum(-1))

    y = torch.stack(outputs, dim=1)
    return y + x * d.view(1, 1, -1)


def selective_scan(x, delta, a, b, c, d):
    if selective_scan_fn is None or x.device.type != "cuda":
        return selective_scan_pytorch(x, delta, a, b, c, d)

    dtype = x.dtype
    u = x.transpose(1, 2).contiguous().to(dtype)
    delta_kernel = delta.transpose(1, 2).contiguous().to(dtype)
    b_kernel = b.permute(0, 2, 1).unsqueeze(1).contiguous().to(dtype)
    c_kernel = c.permute(0, 2, 1).unsqueeze(1).contiguous().to(dtype)
    y = selective_scan_fn(
        u,
        delta_kernel,
        a.float(),
        b_kernel,
        c_kernel,
        d.float(),
        None,
        None,
        False,
    )
    return y.transpose(1, 2)
