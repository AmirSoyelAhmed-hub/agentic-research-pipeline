import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pydantic import ValidationError
from pipelines.models import Paper
from datetime import datetime


def test_valid_paper_passes():
    paper = Paper(
        arxiv_id="2607.12345",
        title="A Great RL Paper",
        abstract="This paper studies something interesting.",
        authors=["Jane Doe"],
        published=datetime.now(),
        url="http://arxiv.org/abs/2607.12345",
    )
    assert paper.title == "A Great RL Paper"


def test_empty_title_rejected():
    with pytest.raises(ValidationError):
        Paper(
            arxiv_id="2607.12345",
            title="   ",
            abstract="Some abstract",
            authors=["Jane Doe"],
            published=datetime.now(),
            url="http://arxiv.org/abs/2607.12345",
        )


def test_no_authors_rejected():
    with pytest.raises(ValidationError):
        Paper(
            arxiv_id="2607.12345",
            title="A Paper",
            abstract="Some abstract",
            authors=[],
            published=datetime.now(),
            url="http://arxiv.org/abs/2607.12345",
        )