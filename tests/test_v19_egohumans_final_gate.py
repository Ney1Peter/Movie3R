from __future__ import annotations

from versions.v19.egohumans.freeze_final_candidate import CORE, action_decision


def _rows(actions: list[str], values: dict[str, float] | None = None) -> list[dict[str, str]]:
    values = values or {}
    rows = []
    for action in actions:
        value = values.get(action, 1.0)
        row = {"case_id": f"{action}-case", "action": action}
        row.update({metric: str(value) for metric in CORE})
        rows.append(row)
    return rows


def test_six_available_actions_with_five_nonworse_passes_action_gate() -> None:
    actions = [f"action-{index}" for index in range(6)]
    result = action_decision(_rows(actions, {actions[-1]: 1.1}), _rows(actions))

    assert result["available_action_count"] == 6
    assert result["nonworse_actions"] == 5
    assert result["at_least_five_available_nonworse"] is True


def test_six_available_actions_with_four_nonworse_fails_action_gate() -> None:
    actions = [f"action-{index}" for index in range(6)]
    result = action_decision(
        _rows(actions, {actions[-2]: 1.1, actions[-1]: 1.1}),
        _rows(actions),
    )

    assert result["nonworse_actions"] == 4
    assert result["at_least_five_available_nonworse"] is False


def test_fewer_than_five_structurally_available_actions_fails_action_gate() -> None:
    actions = [f"action-{index}" for index in range(4)]
    result = action_decision(_rows(actions), _rows(actions))

    assert result["available_action_count"] == 4
    assert result["at_least_five_available_nonworse"] is False


def test_candidate_error_does_not_reduce_available_action_denominator() -> None:
    actions = [f"action-{index}" for index in range(6)]
    result = action_decision(_rows(actions[:4]), _rows(actions))

    assert result["available_action_count"] == 6
    assert result["comparable_candidate_action_count"] == 4
    assert result["candidate_missing_actions"] == actions[4:]
    assert result["at_least_five_available_nonworse"] is False
