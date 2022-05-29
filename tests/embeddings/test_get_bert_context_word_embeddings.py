"""
Partial Tests for the embeddings.get_bert_context_word_embeddings module
"""
import pandas as pd

from alternationprober.embeddings.get_bert_context_word_embeddings import get_sentences


def test_get_sentences():
    """Test that get_sentences() returns only grammatical sentences for the verb."""

    sentence_df = pd.DataFrame.from_dict(
        {
            0: {
                "alternation": "dat",
                "label": 1,
                "sentence": "grammatical sentence verbed the dative guy",
            },
            1: {
                "alternation": "dat",
                "label": 0,
                "sentence": "ungrammatical sentence verbed the dative guy.",
            },
            2: {
                "alternation": "inch",
                "label": 1,
                "sentence": "grammatical sentence verbed the inchoative dog",
            },
            3: {
                "alternation": "inch",
                "label": 0,
                "sentence": "ungrammatical sentence verbed the inchoative fish",
            },
            4: {
                "alternation": "inch",
                "label": 1,
                "sentence": "This sentence nouned, man.",
            },
        },
        orient="index",
    )

    expected_output = ['grammatical sentence verbed the dative guy',
                       'grammatical sentence verbed the inchoative dog']

    computed_output = get_sentences("verbed", sentence_df)

    assert expected_output == computed_output
