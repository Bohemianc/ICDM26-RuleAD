SUPPORTED_RULE_TYPES = {
    "smarts_count",
    "smarts_ratio",
    "descriptor",
    "descriptor_ratio",
    "desirability_window",
    "element_fraction",
}


SUPPORTED_DESCRIPTORS = {
    "mol_wt",
    "logp",
    "tpsa",
    "hba",
    "hbd",
    "rot_bonds",
    "ring_count",
    "aromatic_rings",
    "hetero_atoms",
    "heavy_atoms",
    "fraction_csp3",
    "formal_charge_abs",
    "labute_asa",
    "num_heterocycles",
    "num_bridgehead_atoms",
    "num_spiro_atoms",
    "num_amide_bonds",
    "num_stereo_centers",
    "atom_richness",
    "atom_count",
}


SUPPORTED_NORMALIZERS = SUPPORTED_DESCRIPTORS | {"heavy_atoms", "atom_count"}


SUPPORTED_DIRECTIONS = {"higher", "lower", "two_sided"}


def _validate_structured_rule(rule: dict) -> dict:
    if not isinstance(rule, dict):
        raise ValueError("Each rule must be an object")
    rule_type = rule["type"]
    if rule_type not in SUPPORTED_RULE_TYPES:
        raise ValueError(f"Unsupported structured LLM rule type: {rule_type}")

    direction = rule.get("direction", "higher")
    if direction not in SUPPORTED_DIRECTIONS:
        raise ValueError(f"Unsupported rule direction: {direction}")

    if rule_type == "descriptor":
        if rule["descriptor"] not in SUPPORTED_DESCRIPTORS:
            raise ValueError(f"Unsupported descriptor: {rule['descriptor']}")
    elif rule_type == "descriptor_ratio":
        if rule["numerator"] not in SUPPORTED_NORMALIZERS:
            raise ValueError(f"Unsupported numerator: {rule['numerator']}")
        if rule["denominator"] not in SUPPORTED_NORMALIZERS:
            raise ValueError(f"Unsupported denominator: {rule['denominator']}")
    elif rule_type == "desirability_window":
        if rule["descriptor"] not in SUPPORTED_NORMALIZERS:
            raise ValueError(f"Unsupported desirability descriptor: {rule['descriptor']}")
        floats = [float(rule[key]) for key in ("a", "b", "c", "d")]
        if not (floats[0] <= floats[1] <= floats[2] <= floats[3]):
            raise ValueError("Expected a <= b <= c <= d for desirability_window")
    elif rule_type == "smarts_ratio":
        if rule.get("normalizer", "heavy_atoms") not in SUPPORTED_NORMALIZERS:
            raise ValueError(f"Unsupported SMARTS normalizer: {rule['normalizer']}")
    elif rule_type == "element_fraction":
        if rule.get("normalizer", "heavy_atoms") not in SUPPORTED_NORMALIZERS:
            raise ValueError(f"Unsupported element_fraction normalizer: {rule['normalizer']}")
    return rule


def validate_structured_rule_object(rule: dict) -> dict:
    return _validate_structured_rule(rule)
