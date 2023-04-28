import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sentence_transformers import SentenceTransformer
import docx


class text_analyzer:
    """
    A class that provides methods for text analysis.

    Attributes
    ----------
    model : sentence_transformers.SentenceTransformer
        An instance of the SentenceTransformer model.
    stop_words : set
        A set of stop words for filtering out common words.

    Methods
    -------
    read_word_file(file_path)
        Reads a Word file and returns its text content as a string.
    preprocess_text(text)
        Tokenizes and filters out non-alphanumeric characters from a text string.
    cosine_similarity(vec1, vec2)
        Calculates the cosine similarity between two vectors.
    compare_texts(text1_tokens, text2_tokens, top_n_scores=10)
        Compares the similarity between two sets of text tokens.
    calculate_word_frequency(text_tokens)
        Calculates the frequency distribution of words in a set of text tokens.
    """

    def __init__(self):
        """
        Initializes a new instance of the TextAnalyzer class.
        """
        self.model = SentenceTransformer("bert-base-nli-mean-tokens")
        self.stop_words = set(stopwords.words("english"))

    def read_word_file(self, file_path):
        """
        Reads a Word file and returns its text content as a string.

        Parameters
        ----------
        file_path : str
            The path to the Word file.

        Returns
        -------
        str
            The text content of the Word file.
        """
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text
        return text

    def preprocess_text(self, text):
        """
        Tokenizes and filters out non-alphanumeric characters from a text string.

        Parameters
        ----------
        text : str
            The text string to preprocess.

        Returns
        -------
        list
            A list of tokens from the preprocessed text.
        """
        words = word_tokenize(text.lower())
        words_filtered = [word for word in words if word.isalnum()]
        return words_filtered

    def cosine_similarity(self, vec1, vec2):
        """
        Calculates the cosine similarity between two vectors.

        Parameters
        ----------
        vec1 : numpy.ndarray
            The first vector.
        vec2 : numpy.ndarray
            The second vector.

        Returns
        -------
        float
            The cosine similarity between the two vectors.
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def compare_texts(self, text1_tokens, text2_tokens, top_n_scores=10):
        """
        Compares the similarity between two sets of text tokens.

        Parameters
        ----------
        text1_tokens : list
            The tokens of the first set of text.
        text2_tokens : list
            The tokens of the second set of text.
        top_n_scores : int, optional
            The number of top similarity scores to return (default is 10).

        Returns
        -------
        float
            The similarity score between the two sets of text tokens.
        """
        text1_sentences = " ".join(text1_tokens)
        text2_sentences = " ".join(text2_tokens)

        text1_embedding = self.model.encode(text1_sentences, convert_to_tensor=True)
        text2_embedding = self.model.encode(text2_sentences, convert_to_tensor
