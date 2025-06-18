import pytest
from typing import List, Dict, Any
from haystack import component, Document, Pipeline
from haystack.components.joiners import DocumentJoiner
from dc_custom_component.custom_components.rankers.regex_booster import RegexBooster

# Unit Tests

def test_regex_booster_initialization():
    booster = RegexBooster({"pattern": 1.5})
    assert len(booster.regex_boosts) == 1
    assert list(booster.regex_boosts.values())[0] == 1.5

def test_regex_booster_case_insensitivity():
    booster = RegexBooster({r"\bPython\b": 1.5})
    doc = Document(content="python is great", score=1.0)
    result = booster.run(documents=[doc])
    assert result["documents"][0].score == 1.5

def test_regex_booster_multiple_patterns():
    booster = RegexBooster({r"\bPython\b": 1.5, r"\bgreat\b": 1.2})
    doc = Document(content="Python is great", score=1.0)
    result = booster.run(documents=[doc])
    assert result["documents"][0].score == 1.5 * 1.2

def test_regex_booster_no_match():
    booster = RegexBooster({r"\bJava\b": 1.5})
    doc = Document(content="Python is great", score=1.0)
    result = booster.run(documents=[doc])
    assert result["documents"][0].score == 1.0

def test_regex_booster_sorting():
    booster = RegexBooster({r"\bPython\b": 1.5, r"\bJava\b": 1.2})
    docs = [
        Document(content="Java is okay", score=1.0),
        Document(content="Python is great", score=1.0),
        Document(content="C++ is fast", score=1.0)
    ]
    result = booster.run(documents=docs)
    assert [doc.content for doc in result["documents"]] == ["Python is great", "Java is okay", "C++ is fast"]

def test_regex_booster_no_score():
    booster = RegexBooster({r"\bPython\b": 1.5})
    doc = Document(content="Python is great")
    result = booster.run(documents=[doc])
    assert result["documents"][0].score is None

# Integration Tests

@component
class MockRetriever:
    @component.output_types(documents=List[Document])
    def run(self, query: str) -> Dict[str, Any]:
        docs = [
            Document(content="Python is a programming language", score=0.9),
            Document(content="Java is also a programming language", score=0.7),
            Document(content="Machine learning is a subset of AI", score=0.5)
        ]
        return {"documents": docs}

@pytest.fixture
def regex_pipeline():
    retriever = MockRetriever()
    regex_booster = RegexBooster({r"\bPython\b": 1.5, r"\bAI\b": 1.3})
    joiner = DocumentJoiner()
    
    pipeline = Pipeline()
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("regex_booster", regex_booster)
    pipeline.add_component("joiner", joiner)
    
    pipeline.connect("retriever.documents", "regex_booster.documents")
    pipeline.connect("regex_booster.documents", "joiner.documents")
    
    return pipeline

def test_regex_booster_in_pipeline(regex_pipeline):
    results = regex_pipeline.run(data={"query": "programming languages"})
    documents = results["joiner"]["documents"]
    
    assert len(documents) == 3
    assert documents[0].content == "Python is a programming language"
    assert pytest.approx(documents[0].score, 0.01) == 0.9 * 1.5
    assert documents[1].content == "Java is also a programming language"
    assert pytest.approx(documents[1].score, 0.01) == 0.7
    assert documents[2].content == "Machine learning is a subset of AI"
    assert pytest.approx(documents[2].score, 0.01) == 0.5 * 1.3

def test_regex_booster_pipeline_no_matches():
    @component
    class NoMatchRetriever:
        @component.output_types(documents=List[Document])
        def run(self, query: str) -> Dict[str, Any]:
            return {
                "documents": [
                    Document(content="C++ is a compiled language", score=0.8),
                    Document(content="Ruby is dynamic", score=0.6)
                ]
            }
    
    new_pipeline = Pipeline()
    new_pipeline.add_component("retriever", NoMatchRetriever())
    new_pipeline.add_component("regex_booster", RegexBooster({r"\bPython\b": 1.5, r"\bAI\b": 1.3}))
    new_pipeline.add_component("joiner", DocumentJoiner())
    
    new_pipeline.connect("retriever.documents", "regex_booster.documents")
    new_pipeline.connect("regex_booster.documents", "joiner.documents")
    
    results = new_pipeline.run(data={"query": "programming languages"})
    documents = results["joiner"]["documents"]
    
    assert len(documents) == 2
    assert documents[0].content == "C++ is a compiled language"
    assert pytest.approx(documents[0].score, 0.01) == 0.8
    assert documents[1].content == "Ruby is dynamic"
    assert pytest.approx(documents[1].score, 0.01) == 0.6


