from secondplayer.cli import _parser


def test_bench_reaction_ms_defaults_to_none():
    args = _parser().parse_args(["bench", "tetris", "game.sfc"])
    assert args.reaction_ms is None


def test_bench_reaction_ms_override():
    args = _parser().parse_args(["bench", "tetris", "game.sfc", "--reaction-ms", "200"])
    assert args.reaction_ms == 200.0


def test_bench_reaction_ms_zero_means_uncapped():
    args = _parser().parse_args(["bench", "tetris", "game.sfc", "--reaction-ms", "0"])
    assert args.reaction_ms == 0.0


def test_bench_from_state_defaults_to_cold_boot():
    args = _parser().parse_args(["bench", "tetris", "game.sfc"])
    assert args.from_state is None


def test_bench_from_state_debug_override():
    args = _parser().parse_args(["bench", "tetris", "game.sfc", "--from-state", "debug.state"])
    assert str(args.from_state) == "debug.state"


def test_bench_snap_every_defaults_off():
    args = _parser().parse_args(["bench", "tetris", "game.sfc"])
    assert args.snap_every == 0
    args = _parser().parse_args(["bench", "tetris", "game.sfc", "--snap-every", "50"])
    assert args.snap_every == 50


def test_bench_sample_override_defaults_to_config():
    args = _parser().parse_args(["bench", "tetris", "game.sfc"])
    assert args.sample is None
    args = _parser().parse_args(["bench", "tetris", "game.sfc", "--sample"])
    assert args.sample is True
    args = _parser().parse_args(["bench", "tetris", "game.sfc", "--no-sample"])
    assert args.sample is False
