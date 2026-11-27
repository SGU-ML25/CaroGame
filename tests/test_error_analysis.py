import json
import random

import pytest
from naive import winning_squares as naive_winning_squares

from caro.analysis.error_analysis import (
    LOST,
    analyze_game,
    classify_move,
    double_threats,
    find_games,
    four_squares,
    run_analysis,
    safe_squares,
)
from caro.engine.bitboard import CELLS, bit, index, iter_bits, popcount


def stones(*cells):
    return sum(bit(r, c) for r, c in cells)


def test_missed_win_and_taken_win():
    me = stones((5, 2), (5, 3), (5, 4), (5, 5))
    opp = stones((0, 0), (0, 1))
    assert classify_move(me, opp, index(9, 9))[:2] == ("thang_ngay", True)
    # Hai đầu đều thắng: đi đầu nào cũng đúng, không bị tính là lỗi.
    for end in (index(5, 1), index(5, 6)):
        assert classify_move(me, opp, end)[:2] == ("thang_ngay", False)


def test_block_single_five():
    me = stones((5, 1), (0, 0))
    opp = stones((5, 2), (5, 3), (5, 4), (5, 5))
    assert classify_move(me, opp, index(5, 6))[:2] == ("chan_nam", False)
    assert classify_move(me, opp, index(9, 9))[:2] == ("chan_nam", True)


def test_open_four_is_already_lost():
    opp = stones((5, 2), (5, 3), (5, 4), (5, 5))
    assert classify_move(stones((0, 0)), opp, index(5, 1)) == (LOST, False, 0)


def test_open_three_must_be_answered():
    me = stones((0, 0), (0, 2))
    opp = stones((5, 3), (5, 4), (5, 5))
    assert double_threats(opp, me) == stones((5, 2), (5, 6))
    for cell in ((5, 2), (5, 6)):
        assert classify_move(me, opp, index(*cell))[:2] == ("chan_don_kep", False)
    assert classify_move(me, opp, index(9, 9))[:2] == ("chan_don_kep", True)


def test_two_open_threes_cannot_be_defended():
    me = stones((0, 0), (0, 2))
    opp = stones((3, 3), (3, 4), (3, 5), (7, 3), (7, 4), (7, 5))
    assert classify_move(me, opp, index(3, 2)) == (LOST, False, 0)


def test_own_four_defends_only_if_the_forced_block_is_harmless():
    opp_three = [(5, 3), (5, 4), (5, 5)]
    # Tứ ở cột 0: đối thủ buộc phải chặn tại (5,0), nước chặn đó không tạo gì thêm.
    me = stones((1, 0), (2, 0), (3, 0))
    opp = stones((0, 0), *opp_three)
    assert safe_squares(me, opp, double_threats(opp, me)) >> index(4, 0) & 1
    # Tứ ở cột 6: ô chặn bắt buộc (5,6) lại cho đối thủ tứ hở, nên nước tạo tứ này không đỡ được.
    me = stones((1, 6), (2, 6), (3, 6))
    opp = stones((0, 6), *opp_three)
    assert not safe_squares(me, opp, double_threats(opp, me)) >> index(4, 6) & 1


def test_quiet_position_has_no_category():
    assert classify_move(stones((5, 5)), stones((6, 6)), index(4, 4)) is None


def brute_double_threats(attacker, defender):
    res = 0
    for s in iter_bits(~(attacker | defender) & ((1 << CELLS) - 1)):
        after = attacker | 1 << s
        if popcount(naive_winning_squares(after, defender)) >= 2 and not naive_winning_squares(defender, after):
            res |= 1 << s
    return res


def test_pruned_search_matches_brute_force():
    rng = random.Random(7)
    checked = 0
    while checked < 6:
        cells = rng.sample([index(r, c) for r in range(2, 9) for c in range(2, 9)], 22)
        me, opp = sum(1 << i for i in cells[:11]), sum(1 << i for i in cells[11:])
        if naive_winning_squares(me, opp) or naive_winning_squares(opp, me):
            continue
        threats = brute_double_threats(opp, me)
        assert double_threats(opp, me) == threats
        if not threats:
            continue
        safe = 0
        for m in iter_bits(~(me | opp) & ((1 << CELLS) - 1)):
            if not brute_double_threats(opp, me | 1 << m):
                safe |= 1 << m
        assert safe_squares(me, opp, threats) == safe
        checked += 1


def test_four_squares():
    me = stones((5, 3), (5, 4), (5, 5))
    assert four_squares(me, 0) == stones((5, 1), (5, 2), (5, 6), (5, 7))


def record(moves, **extra):
    return {"game_id": 0, "agent_x": "A", "agent_o": "B", "opening_id": 0, "opening_plies": 0,
            "moves": moves, "result": 1, **extra}


def test_analyze_game_attributes_errors_and_skips_opening():
    # X dựng bốn quân hàng 5, O đi xa; nước thứ 8 O bỏ chặn rồi X thắng.
    moves = [index(5, 2), index(0, 0), index(5, 3), index(0, 2), index(5, 4), index(0, 4), index(5, 5),
             index(10, 10), index(5, 6)]
    counts, errors = analyze_game(record(moves))
    assert [(e["agent"], e["ply"], e["category"]) for e in errors] == [("B", 5, "chan_don_kep")]
    assert counts["B"]["da_thua"] == 1            # sau tứ hở, nước 8 không còn cứu được
    assert counts["A"]["thang_ngay"] == 1 and counts["A"]["thang_ngay_sai"] == 0
    skipped, _ = analyze_game(record(moves, opening_plies=6))
    assert skipped["B"]["moves"] == 1


def test_find_games_prefers_latest_run_with_log(tmp_path):
    for name, text in (("20260101-000000", "{}\n"), ("20260102-000000", "{}\n"), ("20260103-000000", "")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "games.jsonl").write_text(text)
    assert find_games(tmp_path).parent.name == "20260102-000000"
    with pytest.raises(FileNotFoundError):
        find_games(tmp_path / "20260101-000000" / "khong-co")


def test_run_analysis_writes_log(tmp_path):
    source = tmp_path / "giai"
    source.mkdir()
    moves = [index(5, 2), index(0, 0), index(5, 3), index(0, 2), index(5, 4), index(0, 4), index(5, 5),
             index(10, 10), index(5, 6)]
    (source / "games.jsonl").write_text(json.dumps(record(moves)) + "\n")
    out = run_analysis([str(source)], workers=1,
                       overrides=[f"run.root={tmp_path / 'runs'}", "run.index=null", "log.console=false"])
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok" and summary["total_games"] == 1 and summary["total_errors"] == 1
    assert summary["by_agent"]["B"]["chan_don_kep"] == {"chances": 1, "errors": 1}
    assert len((out / "errors.jsonl").read_text(encoding="utf-8").splitlines()) == 1
