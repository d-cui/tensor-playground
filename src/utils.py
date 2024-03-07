from contextlib import contextmanager


@contextmanager
def pretty_print(title: str):
    print(title)
    yield
    print("")  # Add a newline
