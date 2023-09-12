from typing import List, Tuple

from langchain_experimental.language_detector.base import LanguageDetectorBase

try:
    import fasttext
except ImportError:
    raise ImportError(
        "Could not import fasttext, please install with " "`pip install fasttext-wheel`."
    )


class FastTextDetector(LanguageDetectorBase):
    """
    Language detector based on fasttext package.
    It supports 176 languages out of the box.
    """
    
    def __init__(self, threshold: float = 0.1):
        self.model = fasttext.load_model("lid.176.ftz")
        self.threshold = threshold
    
    def _detect_single(self, text: str) -> str:
        """Detects the most probable language of a single text."""
        return self.model.predict(text)[0][0].replace('__label__', '')

    def _detect_many(self, text: str) -> List[Tuple[str, float]]:
        """Detects all languages of a single text with a score bigger than threshold.
        Returns them sorted on score in descending order.
        """
        languages, scores = self.model.predict(text, k=5)
        
        return [
            (lang.replace("__label__", ""), score) 
            for lang, score in zip(languages, scores) if score > self.threshold
        ]
