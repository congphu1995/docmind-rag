from backend.app.pipeline.chunkers.sentence_splitter import split_sentences


def test_simple_sentences():
    text = "Hello world. This is a test. Another sentence here."
    result = split_sentences(text)
    assert result == ["Hello world.", "This is a test.", "Another sentence here."]


def test_abbreviations_not_split():
    text = "Dr. Smith went to Washington D.C. for a meeting. He arrived early."
    result = split_sentences(text)
    # Should not split on Dr. or D.C.
    assert result[-1] == "He arrived early."
    assert len(result) == 2


def test_question_and_exclamation():
    text = "What is this? It is great! And it works."
    result = split_sentences(text)
    assert result == ["What is this?", "It is great!", "And it works."]


def test_newlines_treated_as_boundaries():
    text = "First sentence.\nSecond sentence.\nThird sentence."
    result = split_sentences(text)
    assert len(result) == 3


def test_empty_string():
    text = ""
    result = split_sentences(text)
    assert result == []


def test_single_sentence_no_period():
    text = "Just a fragment"
    result = split_sentences(text)
    assert result == ["Just a fragment"]


def test_multi_line_paragraph():
    text = "This is line one. This is line two.\n\nThis is a new paragraph. With two sentences."
    result = split_sentences(text)
    assert len(result) == 4
