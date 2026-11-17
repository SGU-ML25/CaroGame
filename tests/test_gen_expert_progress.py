"""Định dạng dòng tiến độ của gen_expert_data."""

import time

from caro.training.gen_expert_data import format_duration, format_progress


def test_format_duration():
    assert format_duration(0) == "0s"
    assert format_duration(59.4) == "59s"
    assert format_duration(274) == "4p34s"
    assert format_duration(3909) == "1g05p09s"
    assert format_duration(-5) == "0s"


def test_progress_line_has_percent_bar_rate_and_eta():
    now = time.mktime((2026, 11, 16, 15, 0, 0, 0, 0, -1))
    line = format_progress(done=1000, total=20000, new_done=1000, samples=24000, elapsed=60, now=now, width=10)
    assert "[  5.0%]" in line and "░░░░░░░░░░" in line      # 5% chưa tô ô nào (width 10)
    assert "1000/20000 ván" in line
    assert "16.7 ván/s" in line
    assert "24.000 mẫu" in line
    assert "đã chạy 1p00s" in line
    assert "còn ~19p00s" in line and "xong lúc 15:19" in line


def test_resumed_games_do_not_inflate_speed():
    """959 ván nạp lại từ file không được tính vào tốc độ của lần chạy này."""
    line = format_progress(done=1059, total=2059, new_done=100, samples=0, elapsed=10, now=0)
    assert "10.0 ván/s" in line and "còn ~1p40s" in line


def test_finished_and_no_progress_yet():
    done = format_progress(done=50, total=50, new_done=50, samples=1, elapsed=5, width=4)
    assert "100.0%" in done and "████" in done and "còn" not in done
    start = format_progress(done=0, total=50, new_done=0, samples=0, elapsed=0)
    assert "ván/s" not in start and "còn" not in start
