def train_replay(model, optimizer, replay, settings, device, rng, augment) -> dict:
    """Lấy mẫu đều theo thế cờ trong cửa sổ replay, chỉ ghép mảng của minibatch."""
    games = tuple(replay)
    ends = np.cumsum([len(game[2]) for game in games])
    starts = np.r_[0, ends[:-1]]
    sums = dict.fromkeys(("loss", "policy_loss", "value_loss"), 0.0)
    model.train()
    for _ in range(settings["steps_per_iter"]):
        indices = rng.integers(ends[-1], size=settings["batch_size"])
        game_ids = np.searchsorted(ends, indices, side="right")
        positions = indices - starts[game_ids]
        samples = [tuple(arr[pos] for arr in games[i]) for i, pos in zip(game_ids, positions, strict=True)]
        data = ArrayDataset(*(np.stack(column) for column in zip(*samples, strict=True)), augment=augment)
        states, policy, values = [
            torch.stack(column).to(device) for column in zip(*(data[i] for i in range(len(data))), strict=True)
        ]
        legal = ~states[:, :2].bool().any(dim=1).reshape(-1, N * N)
        logits, pred = model(states, legal)
        log_probs = F.log_softmax(logits, dim=1).masked_fill(~legal, 0.0)
        policy_loss = -(policy * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(pred.squeeze(1), values)
        loss = policy_loss + value_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Loss L8 không hữu hạn.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        for name, value in zip(sums, (loss, policy_loss, value_loss), strict=True):
            sums[name] += value.item()
    return {name: value / settings["steps_per_iter"] for name, value in sums.items()}
