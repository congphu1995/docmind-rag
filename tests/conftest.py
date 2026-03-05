import os

import pytest


@pytest.fixture
def sample_pdf_path():
    path = "tests/fixtures/sample.pdf"
    if not os.path.exists(path):
        pytest.skip("Test PDF not found at tests/fixtures/sample.pdf")
    return path
