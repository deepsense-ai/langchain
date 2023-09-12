from itertools import product
from typing import List, Tuple

import pytest

from langchain_experimental.language_detector.base import LanguageDetectorBase


def _init_detector(model: str) -> LanguageDetectorBase:
    if model == "langdetect":
        from langchain_experimental.language_detector import LangDetector
        return LangDetector()
    if model == "fasttext":
        from langchain_experimental.language_detector import FastTextDetector
        return FastTextDetector()


@pytest.mark.requires("langdetect", "fasttext")
@pytest.mark.parametrize(
    "model,text_language",
    list(
        product(
            ["langdetect", "fasttext"],
            [
                ("Hello, my name is John Doe.", "en"),
                ("Hallo, mein Name ist John Doe.", "de"),
            ],
        )
    )
)
def test_detect_single_language(model: str, text_language: Tuple[str, str]) -> None:
    """Test detecting most probable language of a text"""
    lang_detector = _init_detector(model)
    
    predicted = lang_detector.detect_single_language(text_language[0])
    assert predicted == text_language[1]


@pytest.mark.requires("langdetect", "fasttext")
@pytest.mark.parametrize(
    "model,text_languages",
    list(
        product(
            ["langdetect", "fasttext"],
            [
                ("Hello, my name is John Doe.", ["en"]),
                (
                    "Hello, my name is John Doe. I live in London. Auf Wiedersehen.",
                    ["de", "en"],
                ),
            ]
        )
    )
)
def test_detect_many_languages(model: str, text_languages: Tuple[str, List[str]]) -> None:
    """Test detecting most probable languages of a text"""
    lang_detector = _init_detector(model)
    
    predicted = lang_detector.detect_many_languages(text_languages[0])
    if len(predicted) > 1:
        assert predicted[0][1] > predicted[1][1]  # assert first language is more probable
    assert sorted([x[0] for x in predicted]) == text_languages[1]  # sort languages alphabetically due to randomness of results 
    
