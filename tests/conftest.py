def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: rebuilds a FHIR tokenizer or scans real eval dumps (takes minutes). "
        "Deselect with -m 'not slow'.")
