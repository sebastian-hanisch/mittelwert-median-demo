"""Szenario: Zufallsströme (Block gegen SplitMix64-Klasse), Verteilungen, Reihen, Ebene, Lager."""

import math

import numpy as np
import pytest

import mtw_scenario as S
from mtw_rng import SplitMix64


@pytest.mark.parametrize("seed", [0, 1, 7, 123456789, 2 ** 63 + 5, 2 ** 64 - 1])
def test_block_equals_the_splitmix64_class(seed):
    rng = SplitMix64(seed)
    expected = [rng.next() for _ in range(50)]
    assert [int(v) for v in S.mix_words(seed, 0, 50)] == expected
    assert [int(v) for v in S.mix_words(seed, 20, 10)] == expected[20:30]       # Start mitten im Strom


def test_stream_continues_where_it_stopped():
    a, b = S.Stream(5), S.Stream(5)
    whole = b.words(30)
    parts = np.concatenate([a.words(10), a.words(20)])
    assert np.array_equal(whole, parts) and a.pos == 30


def test_uniform_ranges_and_open_variant():
    u = S.Stream(3).uniform(100_000)
    assert 0.0 <= u.min() and u.max() < 1.0 and abs(u.mean() - 0.5) < 0.005
    v = S.Stream(3).uniform_open(100_000)
    assert 0.0 < v.min() and v.max() < 1.0


def test_normal_is_box_muller_of_the_stream():
    z = S.Stream(11).normal(10)
    rng = SplitMix64(11)
    for i in range(5):
        u1 = ((rng.next() >> 11) + 0.5) * 2.0 ** -53
        u2 = ((rng.next() >> 11) + 0.5) * 2.0 ** -53
        r = math.sqrt(-2.0 * math.log(u1))
        assert z[2 * i] == pytest.approx(r * math.cos(2 * math.pi * u2), abs=1e-12)
        assert z[2 * i + 1] == pytest.approx(r * math.sin(2 * math.pi * u2), abs=1e-12)


def test_normal_moments_and_prefix_property():
    z = S.Stream(7).normal(400_000)
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1.0) < 0.01
    assert abs(((z - z.mean()) ** 3).mean()) < 0.03 and abs(((z - z.mean()) ** 4).mean() - 3.0) < 0.06
    assert np.array_equal(S.Stream(7).normal(1001), z[:1001])             # ungerade Anzahl: der letzte Wert des Paars entfällt
    assert np.array_equal(S.Stream(7).normal(1000), z[:1000])


def test_cell_seed_is_stable_and_separates_cells():
    assert S.cell_seed(11, "e1", "float32") == S.cell_seed(11, "e1", "float32")
    values = {S.cell_seed(11, "e1", "float32"), S.cell_seed(12, "e1", "float32"), S.cell_seed(11, "e1", "float64"), S.cell_seed(11, "e2", "float32"), S.cell_seed(11, 0.8), S.cell_seed(11, 0.9)}
    assert len(values) == 6
    assert S.cell_seed(11, "e1") == 11 * 1_000_003 + __import__("zlib").crc32(b"e1")


def test_distribution_functions_match_scipy():
    st = pytest.importorskip("scipy.stats")
    count = 100_000
    cases = {"Normal": st.norm(), "Laplace": st.laplace(), "Logistisch": st.logistic(), "Gleichverteilung": st.uniform(-1, 2), "t(3)": st.t(3), "t(5)": st.t(5),
             "Exponential": st.expon(), "Pareto(2,5)": st.pareto(2.5)}
    for dist, ref in cases.items():
        x = np.sort(S.draw(dist, S.Stream(17), count))
        ks = np.abs(ref.cdf(x) - (np.arange(1, count + 1) / count)).max()
        assert ks < 0.008, (dist, ks)
    x = np.sort(S.draw("Lognormal", S.Stream(17), count, sigma=0.8))
    assert np.abs(st.lognorm(0.8).cdf(x) - np.arange(1, count + 1) / count).max() < 0.008


def test_poisson_matches_scipy_pmf_and_table_ends_at_one():
    st = pytest.importorskip("scipy.stats")
    for lam in (0.7, 2.0, 10.0):
        x = S.draw("Poisson", S.Stream(5), 200_000, lam=lam)
        assert x.min() >= 0 and np.array_equal(x, np.round(x))
        pmf = np.bincount(x.astype(int)) / len(x)
        assert np.abs(pmf - st.poisson(lam).pmf(np.arange(len(pmf)))).max() < 0.005
        assert S.poisson_table(lam)[-1] == 1.0


def test_unknown_distribution_raises():
    with pytest.raises(ValueError):
        S.draw("Cauchy", S.Stream(1), 5)


def test_rounded_raster():
    assert S.rounded([0.2, 0.3, 0.74, 0.76, -0.3], 0.5).tolist() == [0.0, 0.5, 0.5, 1.0, -0.5]
    assert S.rounded([0.1234], 0.0).tolist() == [0.1234]
    assert S.rounded([1.5, 2.5, 0.5], 1.0).tolist() == [2.0, 3.0, 1.0]       # in der Mitte zweier Rasterpunkte wird aufgerundet


def test_series_is_stored_in_the_format_and_deterministic():
    a = S.make_series(1000, 100.0, "Normal", "zufällig", 3, "float32")
    assert a.dtype == np.float32 and a.shape == (1000,)
    assert np.array_equal(a, S.make_series(1000, 100.0, "Normal", "zufällig", 3, "float32"))
    assert not np.array_equal(a, S.make_series(1000, 100.0, "Normal", "zufällig", 4, "float32"))
    assert S.make_series(10, 1.0, "Normal", "zufällig", 3, "float64").dtype == np.float64
    assert np.array_equal(S.make_series(500, 1000.0, "Normal", "zufällig", 3, "float64"), S.make_series(2000, 1000.0, "Normal", "zufällig", 3, "float64")[:500])   # Präfix


def test_series_mean_and_spread_follow_ratio():
    for dist in S.SERIES_DISTRIBUTIONS:
        x = S.make_series(200_000, 50.0, dist, "zufällig", 9, "float64")
        assert abs(x.mean() - 50.0) < 0.02 and abs(x.std() - 1.0) < 0.05, dist
    skew = lambda v: ((v - v.mean()) ** 3).mean() / v.std() ** 3
    assert skew(S.make_series(200_000, 50.0, "Lognormal", "zufällig", 9, "float64")) > 3.0     # schief: lognormal(0; 1) hat Schiefe 6,18
    assert abs(skew(S.make_series(200_000, 50.0, "Normal", "zufällig", 9, "float64"))) < 0.05


def test_series_orders_are_permutations_of_the_same_values():
    r = S.make_series(2000, 1000.0, "Normal", "zufällig", 5, "float32")
    up = S.make_series(2000, 1000.0, "Normal", "aufsteigend", 5, "float32")
    down = S.make_series(2000, 1000.0, "Normal", "absteigend", 5, "float32")
    assert np.all(np.diff(up) >= 0) and np.all(np.diff(down) <= 0) and np.array_equal(np.sort(r), up) and np.array_equal(up[::-1], down)
    assert not np.array_equal(r, up)


def test_series_rejects_unknown_arguments():
    with pytest.raises(ValueError):
        S.make_series(10, 1.0, "Laplace", "zufällig", 1, "float32")
    with pytest.raises(ValueError):
        S.make_series(10, 1.0, "Normal", "sortiert", 1, "float32")


def test_float32_series_with_huge_ratio_is_coarsely_rounded():
    x = S.make_series(100_000, 1e6, "Normal", "zufällig", 5, "float32")
    assert len(np.unique(x)) < 200                 # Abstand der float32-Zahlen bei 10⁶ ist 0,0625: wenige verschiedene Werte
    assert len(np.unique(S.make_series(100_000, 1e6, "Normal", "zufällig", 5, "float64"))) > 99_000


def test_plane_and_miscount():
    p = S.make_plane(5000, 4)
    assert p.shape == (5000, 2) and abs(p.mean()) < 0.05 and abs(p.std() - 1.0) < 0.05
    q = S.with_miscount(p[:10], 7.0)
    assert q.shape == (11, 2) and q[-1].tolist() == pytest.approx([7 / math.sqrt(2)] * 2) and np.linalg.norm(q[-1]) == pytest.approx(7.0)
    assert np.array_equal(q[:10], p[:10])


@pytest.mark.parametrize("mode", ["eine", "gleich"])
def test_warehouse_truth_has_the_requested_norm(mode):
    for d in (3, 10, 50):
        assert np.linalg.norm(S.warehouse_truth(d, 4.0, mode)) == pytest.approx(4.0)
    mu, X = S.warehouse_observations(5, 2.0, mode, 20_000, S.Stream(3))
    assert X.shape == (20_000, 5) and np.abs(X.mean(axis=0) - mu).max() < 0.03 and np.abs(X.std(axis=0) - 1.0).max() < 0.03


def test_warehouse_mode_must_be_known():
    with pytest.raises(ValueError):
        S.warehouse_truth(3, 1.0, "viele")
