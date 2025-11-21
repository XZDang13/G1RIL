import gymnasium

gymnasium.register(
    id="AMPWalk-v0",
    entry_point=f"{__name__}.amp_env:G1WalkEnv"
)