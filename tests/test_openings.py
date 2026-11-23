
import json

import pytest

from caro.arena.openings import (
    generate_openings,
    load_openings,
)


def test_generate_requested_number_of_openings():
    openings = generate_openings(
        num_openings=5,
        plies=[2, 2],
        region=5,
        seed=7,
    )

    assert len(openings) == 5
    assert all(len(opening) == 2 for opening in openings)


def test_generated_actions_are_inside_center_region():
    openings = generate_openings(
        num_openings=20,
        plies=[2, 4],
        region=5,
        seed=7,
    )

    valid_actions = {
        r * 11 + c
        for r in range(3, 8)
        for c in range(3, 8)
    }

    for opening in openings:
        assert all(action in valid_actions for action in opening)


def test_no_duplicate_action_inside_an_opening():
    openings = generate_openings(
        num_openings=30,
        plies=[2, 4],
        region=5,
        seed=7,
    )

    for opening in openings:
        assert len(opening) == len(set(opening))


def test_same_seed_generates_same_openings():
    first = generate_openings(
        num_openings=20,
        plies=[2, 4],
        region=5,
        seed=123,
    )
    second = generate_openings(
        num_openings=20,
        plies=[2, 4],
        region=5,
        seed=123,
    )

    assert first == second


def test_different_seed_generates_different_openings():
    first = generate_openings(
        num_openings=20,
        plies=[2, 4],
        region=5,
        seed=123,
    )
    second = generate_openings(
        num_openings=20,
        plies=[2, 4],
        region=5,
        seed=456,
    )

    assert first != second


def test_generate_openings_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        generate_openings(num_openings=-1)

    with pytest.raises(ValueError):
        generate_openings(region=12)

    with pytest.raises(ValueError):
        generate_openings(plies=[4, 2])

    with pytest.raises(ValueError):
        generate_openings(plies=[2, 26], region=5)


def test_load_openings_without_config_keeps_legacy_behavior(
    tmp_path, monkeypatch
):
    legacy_file = tmp_path / "openings_v1.json"
    expected = [[36, 37], [48, 49, 50]]

    legacy_file.write_text(
        json.dumps(expected),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "caro.arena.openings.DEFAULT_OPENINGS_FILE",
        legacy_file,
    )

    assert load_openings() == expected


def test_load_openings_without_config_generates_when_file_missing(
    tmp_path, monkeypatch
):
    missing_file = tmp_path / "missing.json"

    monkeypatch.setattr(
        "caro.arena.openings.DEFAULT_OPENINGS_FILE",
        missing_file,
    )

    openings = load_openings()

    assert len(openings) == 100
    assert not missing_file.exists()


def test_file_none_generates_without_writing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    config = {
        "count": 5,
        "plies": [2, 2],
        "region": 5,
        "seed": 7,
        "file": None,
    }

    openings = load_openings(config)

    assert len(openings) == 5
    assert all(len(opening) == 2 for opening in openings)
    assert list(tmp_path.rglob("*.json")) == []


def test_load_openings_creates_and_reuses_configured_file(tmp_path):
    output_file = tmp_path / "openings" / "test.json"

    config = {
        "count": 5,
        "plies": [2, 2],
        "region": 5,
        "seed": 7,
        "file": str(output_file),
    }

    first = load_openings(config)

    assert output_file.exists()
    assert json.loads(output_file.read_text(encoding="utf-8")) == first

    second_config = {
        **config,
        "count": 10,
        "seed": 999,
    }
    second = load_openings(second_config)

    assert second == first
    assert len(second) == 5
