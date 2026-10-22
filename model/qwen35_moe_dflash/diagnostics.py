"""Keep dependency root causes visible without importing model frameworks."""


def format_error(error):
    messages, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        messages.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return " <- ".join(messages)
