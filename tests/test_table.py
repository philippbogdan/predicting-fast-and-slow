"""The table does what the paper says it does, on cases worked by hand."""
import numpy as np

from admission import tracker as trk


def cell(size_class, load_class, age_class, quiet):
    return ((size_class * trk.N_LOAD + load_class) * trk.N_AGE + age_class) * trk.N_QUIET + quiet


def test_table_has_4096_cells_of_one_sixty_fourth():
    assert trk.N_CELLS == 32 * 16 * 4 * 2 == 4096
    assert trk.ALPHA == 1 / 64


def test_cell_of_a_read():
    T = 100.0
    # 4 KiB, nothing outstanding, a completion just now: size class 12, load 0, age 0, not quiet
    assert trk.cell_of(4096, 0, 0.0, 10.0, T) == cell(12, 0, 0, 0)
    # 64 KiB behind 300 KiB, oldest request 250 us old, last completion 2 ms ago: load is the bit length of
    # floor(307200 / 4096) = 75, which is 7; age is floor(250 / 100) = 2; quiet because 2000 > 16 * 100
    assert trk.cell_of(65536, 307200, 250.0, 2000.0, T) == cell(16, 7, 2, 1)
    # nothing has completed yet (negative time since the last completion): never quiet
    assert trk.cell_of(65536, 307200, 250.0, -1.0, T) == cell(16, 7, 2, 0)
    # the classes are capped
    assert trk.cell_of(2.0 ** 40, 2.0 ** 60, 1e9, 10.0, T) == cell(31, 15, 3, 0)


def test_an_entry_moves_one_sixty_fourth_of_the_way_and_only_after_completion():
    T = 100.0
    # four reads of one cell: submitted at 0, 10, 1000 and 2000 us; the first two take 500 us (slow), the third 50
    ts = np.array([0.0, 10.0, 1000.0, 2000.0])
    lat = np.array([500.0, 500.0, 50.0, 50.0])
    q = trk.evidence(ts, lat, size=np.full(4, 4096.0), out_bytes=np.zeros(4), oldest_age=np.zeros(4),
                     since_done=np.full(4, -1.0), T=T)
    assert q[0] == 0.5                      # nothing has completed
    assert q[1] == 0.5                      # the first read is still in flight
    after_two_slow = 0.5 + (1 - 0.5) / 64
    after_two_slow += (1 - after_two_slow) / 64
    assert np.isclose(q[2], after_two_slow)
    assert np.isclose(q[3], after_two_slow + (0 - after_two_slow) / 64)


def test_the_score_is_q_against_a_fast_guess_and_one_minus_q_against_a_slow_one():
    q = np.array([0.9, 0.9])
    said_slow = np.array([False, True])
    assert np.allclose(trk.mistake_score(q, said_slow), [0.9, 0.1])
