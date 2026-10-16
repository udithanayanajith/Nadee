"""Features for worker-operation productivity prediction and line planning."""

WORKER_OPERATION_FEATURES = [
    "grade_rank", "experience_months", "is_trained", "attendance_rate",
    "worker_efficiency", "worker_quality", "worker_rework_rate",
    "previous_output", "operation_smv", "operation_sequence",
    "operation_complexity", "operation_quality_requirement",
    "operation_category", "machine_type", "skill_count", "skill_match",
]

GRADE_RANK = {"C": 0, "B": 1, "A": 2, "A+": 3, "Jumper": 4}
CATEGORY_RANK = {"Basic": 0, "Intermediate": 1, "Critical": 2}
MACHINE_TYPES = {
    name: index for index, name in enumerate(
        ["Bartack", "Buttonhole", "Flatlock", "Kansai", "Overlock", "SNLS"]
    )
}


def make_worker_operation_features(worker, operation):
    skills = {
        skill.strip().casefold()
        for skill in str(worker.get("machine_skills", "")).split(",")
        if skill.strip()
    }
    required_machine = str(operation.get("machine_type_required", "")).strip()
    return {
        "grade_rank": float(GRADE_RANK.get(str(worker.get("grade", "C")), 0)),
        "experience_months": float(worker.get("experience_months", 0) or 0),
        "is_trained": float(str(worker.get("training_status", "")).casefold() == "trained"),
        "attendance_rate": float(worker.get("attendance_rate", 0) or 0),
        "worker_efficiency": float(worker.get("efficiency_avg", 70) or 70),
        "worker_quality": float(worker.get("quality_avg", 85) or 85),
        "worker_rework_rate": float(worker.get("rework_rate", 0) or 0),
        "previous_output": float(worker.get("previous_output", 0) or 0),
        "operation_smv": float(operation.get("smv", 0) or 0),
        "operation_sequence": float(operation.get("sequence_no", 0) or 0),
        "operation_complexity": float(operation.get("complexity_score", 0) or 0),
        "operation_quality_requirement": float(operation.get("quality_requirement", 0) or 0),
        "operation_category": float(CATEGORY_RANK.get(str(operation.get("operation_category", "Basic")), 0)),
        "machine_type": float(MACHINE_TYPES.get(required_machine, -1)),
        "skill_count": float(len(skills)),
        "skill_match": float(required_machine.casefold() in skills),
    }
