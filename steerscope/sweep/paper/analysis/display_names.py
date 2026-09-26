"""Publication names; internal method identifiers remain unchanged."""

METHOD_DISPLAY_NAMES = {
    "PromptSteering": "Prompt Steering",
    "SimplePromptSteering": "Simple Prompt Steering",
    "DiffMean": "DiffMean", "PCA": "PCA", "LAT": "LAT", "Random": "Random",
    "LinearProbe": "Linear Probe", "SteeringVector": "SSV", "LsReFT": "ReFT-r1",
    "GemmaScopeSAE": "SAE", "GemmaScopeSAEMaxAUC": "SAE-A",
    "SphericalSteering": "Spherical Steering", "HiDRA": "HiDRA",
    "AUSteer": "AUSteer", "LoRA": "LoRA", "LoReFT": "LoReFT", "SFT": "SFT",
    "PreferenceVector": "RePS", "HyperSteer": "HyperSteer", "FLAS": "FLAS",
    "ODESteer": "ODESteer", "StepODESteer": "StepODESteer",
    "APSR": "A-PSR", "SPSR": "S-PSR",
}


def method_display_name(method):
    return METHOD_DISPLAY_NAMES.get(str(method), str(method))


def display_method_names(value):
    """Format a DataFrame/Styler for display without mutating analysis data."""
    import copy
    import pandas as pd
    from pandas.io.formats.style import Styler

    if isinstance(value, Styler):
        result = copy.deepcopy(value)
        if "method" in result.data.columns:
            result.format({"method": method_display_name})
        if "method" in result.data.index.names:
            result.format_index(method_display_name, axis=0, level="method")
        return result
    if isinstance(value, pd.DataFrame):
        result = value.copy()
        if "method" in result.columns:
            result["method"] = result["method"].map(method_display_name)
        if "method" in result.index.names:
            result = result.rename(index=method_display_name, level="method")
        if "method" in result.columns.names:
            result = result.rename(columns=method_display_name, level="method")
        return result
    return value
