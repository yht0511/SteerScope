"""Answer extraction adapted from the official hendrycks/math evaluator."""


def last_boxed_only_string(string):
    """Return the last balanced ``\\boxed{...}`` or ``\\fbox{...}``."""
    boxed_index = string.rfind("\\boxed")
    fbox_index = string.rfind("\\fbox")
    index = max(boxed_index, fbox_index)
    if index < 0:
        return None
    depth = 0
    saw_left_brace = False
    for position in range(index, len(string)):
        character = string[position]
        if character == "{":
            depth += 1
            saw_left_brace = True
        elif character == "}" and saw_left_brace:
            depth -= 1
            if depth == 0:
                return string[index:position + 1]
    return None


def remove_boxed(string):
    """Remove the outer official answer command from an extracted answer."""
    if string is None:
        return None
    for prefix in ("\\boxed{", "\\fbox{"):
        if string.startswith(prefix) and string.endswith("}"):
            return string[len(prefix):-1]
    return None
