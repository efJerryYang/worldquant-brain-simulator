import numpy as np

from wqsim.alphas import ALPHAS, AlphaContext


def test_registered_alphas_evaluate_to_input_shape():
    rows = 80
    cols = 5
    base = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols) + 100.0
    context = AlphaContext(
        open=base,
        high=base + 2.0,
        low=base - 2.0,
        close=base + 1.0,
        volume=(base * 1000.0) + 1.0,
        returns=np.sin(base / 10.0),
        vwap=base + 0.5,
    )

    expected = {
        "eg_alpha",
        "eg_alpha2",
        "eg_alpha3",
        "alpha001",
        "alpha002",
        "alpha003",
        "alpha004",
        "alpha005",
        "alpha006",
        "alpha007",
        "alpha008",
        "alpha009",
        "alpha010",
    }

    assert expected == set(ALPHAS)
    for alpha in ALPHAS.values():
        actual = alpha(context)
        assert actual.shape == (rows, cols)
