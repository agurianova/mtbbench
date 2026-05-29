import re
from typing import Any


def _parse_t_stage(pT_stage: str) -> int | None:
    if not pT_stage or not isinstance(pT_stage, str):
        return None
    match = re.search(r"pT(\d+)", pT_stage, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_n_stage(pN_stage: str) -> int | None:
    if not pN_stage or not isinstance(pN_stage, str):
        return None
    pN_stage = pN_stage.upper()
    if pN_stage in ("PNX", "NX", "PN0", "N0"):
        return 0
    match = re.search(r"PN(\d+)", pN_stage, re.IGNORECASE)
    if match:
        return int(match.group(1))
    if "N2" in pN_stage:
        return 2
    if "N1" in pN_stage:
        return 1
    if "N3" in pN_stage:
        return 3
    return None


def _normalize_site(primary_tumor_site: str) -> str:
    if not primary_tumor_site:
        return "other"
    site = primary_tumor_site.lower().replace(" ", "_")
    if "oropharynx" in site:
        return "oropharynx"
    if "oral" in site:
        return "oral_cavity"
    if "larynx" in site:
        return "larynx"
    if "hypopharynx" in site:
        return "hypopharynx"
    return "other"


def _normalize_hpv(hpv_status: str) -> str:
    if not hpv_status:
        return "hpv_unknown"
    status = str(hpv_status).lower()
    if status in ("positive", "pos", "yes"):
        return "hpv_positive"
    if status in ("negative", "neg", "no"):
        return "hpv_negative"
    return "hpv_unknown"


def compute_ajcc_stage(pathological_data: dict[str, Any]) -> str:
    """
    Map pathological pT/pN to a simplified AJCC stage group (I-IV) for cohort lookup.
    """
    t = _parse_t_stage(pathological_data.get("pT_stage", ""))
    n = _parse_n_stage(pathological_data.get("pN_stage", ""))
    site = _normalize_site(pathological_data.get("primary_tumor_site", ""))
    hpv = _normalize_hpv(pathological_data.get("hpv_association_p16", ""))

    if t is None and n is None:
        return "stage_III"

    if site == "oropharynx" and hpv == "hpv_positive":
        if t is not None and t <= 2 and (n is None or n == 0):
            return "stage_I" if t == 1 else "stage_II"
        if t is not None and t <= 2 and n == 1:
            return "stage_III"
        return "stage_IV"

    if t is not None and t == 1 and (n is None or n == 0):
        return "stage_I"
    if t is not None and t == 2 and (n is None or n == 0):
        return "stage_II"
    if (t is not None and t <= 2 and n == 1) or (t is not None and t == 3 and (n is None or n == 0)):
        return "stage_III"
    return "stage_IV"


def stage_patient(pathological_data: dict[str, Any], clinical_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Build a staging summary for logging and survival cohort matching.
    """
    site = _normalize_site(pathological_data.get("primary_tumor_site", ""))
    hpv = _normalize_hpv(pathological_data.get("hpv_association_p16", ""))
    ajcc_stage = compute_ajcc_stage(pathological_data)

    staging = {
        "primary_tumor_site": pathological_data.get("primary_tumor_site"),
        "pT_stage": pathological_data.get("pT_stage"),
        "pN_stage": pathological_data.get("pN_stage"),
        "hpv_status": hpv,
        "ajcc_stage_group": ajcc_stage,
        "histologic_type": pathological_data.get("histologic_type"),
    }
    if clinical_data:
        staging["age_at_diagnosis"] = clinical_data.get("age_at_initial_diagnosis")
        staging["sex"] = clinical_data.get("sex")
    return staging
