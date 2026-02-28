import jax


def pytest_sessionstart(session):
    """Enable JAX 64-bit mode at the start of the pytest session."""
    jax.config.update("jax_enable_x64", True)
