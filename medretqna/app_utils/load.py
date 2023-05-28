import pickle

from mrq.algs import GoldenRetriever
from mrq.models import EmbedModel, NERClassifier
from mrq.logger import get_logger

log = get_logger(__name__)


def load_data_from_pickle(path: str):
    """Load AnswerDB data from pickle"""
    with open(path, "rb") as file:
        return pickle.load(file)


def load_on_startup(config: dict):
    """Load necessary classes for predictions"""
    log.info("Loading NER")
    ner = NERClassifier(config["ner_name"])
    log.info("Loading Embeddings")
    emb = EmbedModel(config["emb"])
    log.info("Loading data")
    db = load_data_from_pickle(config["db"])
    log.info("Instantiating retriever")
    retriever = GoldenRetriever(db, output_format="list")
    log.info("Done.")
    return ner, emb, db, retriever
