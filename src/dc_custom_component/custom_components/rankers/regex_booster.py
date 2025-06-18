import re
from typing import Dict, List
from haystack import component, Document


@component
class RegexBooster:
    r"""
    A component for boosting document scores based on regex patterns.

    This component adjusts the scores of documents based on whether their content
    matches specified regular expression patterns. After adjusting scores, it
    sorts the documents in descending order of their new scores.

    Note:
        - Regex matching is case-insensitive by default.
        - Multiple regex patterns can match a single document, in which case
          the boosts are multiplied together.
        - Documents that don't match any patterns keep their original score.
        - The component assumes documents already have a 'score' attribute.
          Documents without a score are treated as having a score of 0.

    Example:
        ```python
        booster = RegexBooster({
            r"\bpython\b": 1.5,       # Boost documents mentioning "python" by 50%
            r"machine\s+learning": 1.3,  # Boost "machine learning" by 30%
            r"\bsql\b": 0.8,          # Reduce score for documents mentioning "sql" by 20%
        })
        ```
    In this example, a document containing both "python" and "machine learning"
    would have its score multiplied by 1.5 * 1.3 = 1.95, effectively boosting
    it by 95%.
    """

    def __init__(self, regex_boosts: Dict[str, float]):
        self.regex_boosts = {
            re.compile(k, re.IGNORECASE): v for k, v in regex_boosts.items()
        }
        """
        Initialize the component.
        
        :param regex_boosts: A dictionary where:
            - Keys are string representations of regular expression patterns.
            - Values are float numbers representing the boost factor.

            The boost factor must be greater than 1.0 to increase the score,
            or between 0 and 1 to decrease it. A boost of exactly 1.0 will
            have no effect.
        """

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Apply regex-based score boosting to the input documents.

        :param documents: The list of documents to process.

        Returns: A dictionary with a single key 'documents',
            containing the list of processed documents, sorted by their new scores.
        """
        for regex, boost in self.regex_boosts.items():
            for doc in documents:
                if doc.score is not None and regex.search(doc.content):
                    doc.score *= boost

        documents = sorted(documents, key=lambda x: x.score or 0, reverse=True)

        return {"documents": documents}
