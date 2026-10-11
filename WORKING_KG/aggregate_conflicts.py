from __future__ import annotations

from collections.abc import Mapping


def check_grade_capacity(
    total_headcount: Mapping[str, int],
    absent_headcount: Mapping[str, int],
    already_committed: Mapping[str, int],
    incoming_demand: Mapping[str, int],
) -> dict[str, object]:
    """Check aggregate grade-level capacity without inventing worker identities.

    This only finds counts that exceed availability. It cannot tell whether a specific
    person is double-booked; that requires pseudonymous worker IDs and time windows.
    """
    grades = sorted(set(total_headcount) | set(absent_headcount) | set(already_committed) | set(incoming_demand))
    by_grade: dict[str, dict[str, int | bool]] = {}
    for grade in grades:
        total = max(0, int(total_headcount.get(grade, 0)))
        absent = max(0, int(absent_headcount.get(grade, 0)))
        committed = max(0, int(already_committed.get(grade, 0)))
        demand = max(0, int(incoming_demand.get(grade, 0)))
        available = max(0, total - absent - committed)
        shortfall = max(0, demand - available)
        by_grade[grade] = {
            "available": available,
            "incoming_demand": demand,
            "shortfall": shortfall,
            "feasible": shortfall == 0,
        }
    return {"feasible": all(row["feasible"] for row in by_grade.values()), "by_grade": by_grade,
            "conflict_type": "aggregate_grade_capacity_only"}
