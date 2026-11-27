@dataclass
class SelfplayTrace:
    seed: int
    game: CaroGame = field(default_factory=CaroGame)
    states: list[np.ndarray] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    sims: list[int] = field(default_factory=list)

    def step(self, move: int, agent: PUCTAgent) -> None:
        self.states.append(encode(self.game).astype(np.uint8))
        self.policies.append(agent.last_pi)
        stats = agent.last_stats()
        self.times.append(stats["time_ms"])
        self.sims.append(stats["sims"])
        self.game.step(move)

    def finish(self) -> tuple[GameSamples, dict]:
        result = self.game.result()
        values = np.full(len(self.states), result, dtype=np.int8)
        values[1::2] *= -1  # tự đấu luôn bắt đầu từ bàn trống, X đi trước
        return (np.stack(self.states), np.stack(self.policies), values), {
            "seed": self.seed, "moves": self.game.history, "result": result, "termination": self.game.termination(),
            "time_ms": self.times, "sims": self.sims,
        }


def selfplay_games(
    agents: Sequence[PUCTAgent], seeds: Sequence[int], temperature_moves: int,
) -> list[tuple[GameSamples, dict]]:
    traces = [SelfplayTrace(seed) for seed in seeds]
    active = list(zip(agents, traces, strict=True))
    for agent, trace in active:
        agent.reset(trace.seed)
    while active:
        for agent, trace in active:
            agent.temperature = float(len(trace.game.history) < temperature_moves)
        moves = select_actions([agent for agent, _ in active], [trace.game for _, trace in active])
        for (agent, trace), move in zip(active, moves, strict=True):
            trace.step(move, agent)
        active = [(agent, trace) for agent, trace in active if not trace.game.is_terminal()]
    return [trace.finish() for trace in traces]
