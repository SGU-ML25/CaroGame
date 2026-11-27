def update(
    model: CaroResNet, optimizer: torch.optim.Optimizer, batch: dict, temperature: float, entropy_coef: float,
    value_coef: float, grad_clip: float | None, device: torch.device, chunk: int = 1024,
) -> dict:
    """Một bước cập nhật trên toàn bộ mẫu của vòng lặp; chia lô con `chunk` mẫu (cộng dồn gradient) để vừa VRAM."""
    model.train()
    n = len(batch["z"])
    optimizer.zero_grad()
    sums = dict.fromkeys(("loss", "policy_loss", "value_loss", "entropy"), 0.0)
    for lo in range(0, n, chunk):
        sl = slice(lo, lo + chunk)
        x = torch.from_numpy(batch["x"][sl]).to(device)
        mask = torch.from_numpy(batch["mask"][sl]).to(device)
        action = torch.from_numpy(batch["action"][sl]).to(device)
        z = torch.from_numpy(batch["z"][sl]).to(device)

        logits, v = model(x, mask)
        v = v.squeeze(1)
        logp = F.log_softmax(logits / temperature, dim=1)
        chosen = logp.gather(1, action[:, None]).squeeze(1)
        advantage = z - v.detach()                          # baseline: value head, không chảy gradient vào policy
        policy_loss = -(advantage * chosen).mean()
        value_loss = F.mse_loss(v, z)
        entropy = -(logp.exp() * logp.masked_fill(~mask, 0.0)).sum(dim=1).mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        w = len(z) / n                                      # trung bình theo mẫu, không theo lô con
        (loss * w).backward()
        for k, val in (("loss", loss), ("policy_loss", policy_loss), ("value_loss", value_loss), ("entropy", entropy)):
            sums[k] += val.item() * w
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip if grad_clip else float("inf")))
    optimizer.step()
    return {**sums, "grad_norm": grad_norm, "mean_z": float(batch["z"].mean())}
