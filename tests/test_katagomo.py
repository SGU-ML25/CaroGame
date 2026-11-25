"""Chuyển toạ độ và đọc thông tin tìm kiếm của tác tử KataGomo (không cần engine)."""

import pytest

from caro.agents.katagomo import CHAT_RE, KataGomoAgent


def test_gtp_coordinates_round_trip():
    for action in range(121):
        assert KataGomoAgent._action(KataGomoAgent._gtp(action)) == action


@pytest.mark.parametrize(("action", "text"), [
    (0, "A11"),       # hàng trên cùng, cột đầu
    (10, "L11"),      # hàng trên cùng, cột cuối (bỏ I)
    (60, "F6"),       # giữa bàn
    (120, "L1"),      # hàng cuối, cột cuối
])
def test_gtp_coordinates_known_points(action, text):
    assert KataGomoAgent._gtp(action) == text


@pytest.mark.parametrize("reply", ["pass", "PASS", "resign", "", "Z9", "A0"])
def test_invalid_or_pass_reply_has_no_action(reply):
    assert KataGomoAgent._action(reply) is None


def test_chat_parser():
    line = "CHAT:Visits 801 Winrate 79.90% Drawrate 32.06% PV F6 G6 G7 E5 F8"
    match = CHAT_RE.search(line)
    assert match is not None
    assert int(match.group(1)) == 801
    assert float(match.group(2)) == pytest.approx(79.90)
    assert [KataGomoAgent._action(m) for m in match.group(4).split()] == [60, 61, 50, 70, 38]
