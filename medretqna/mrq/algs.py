from typing import Dict

import numpy as np
import torch
from mrq.data import AnswerDB, Query
from mrq.logger import get_logger

log = get_logger(__name__)


class GoldenRetriever:
    def __init__(self, data: AnswerDB) -> None:
        """Retrieval algorithm for most related (golden) db answers to a query

        And a dog;)

        Args:
            data (AnswerDB): data to search in.
        """
        self.data = data
        if not hasattr(self.data, "embedded"):
            log.warning("Data for retrieval task doesn't have embeddings")

    def find_it(
        self, q: Query, top_k: int = None, sorted: bool = True
    ) -> Dict[str, np.array]:
        """Search for most related answers

        Args:
            q (Query): query.
            top_k (int, optional): returns k most relevant records.
                If None returns whole data: Defaults to None.
            sorted (bool, optional): whether to sort data before return.
                Defaults to True.

        Raises:
            ValueError: if top_k is <1

        Returns:
            dict: answers and similarities
        """
        if not top_k:
            top_k = self.data.embedded.shape[0]
        if top_k < 1:
            raise ValueError("top_k should be >=1")
        similarity = self.data.embedded @ q.embedded.T
        similarity = torch.topk(similarity.squeeze(-1).cpu(), k=top_k, sorted=sorted)
        return {
            "answers": self.data.data[similarity.indices],
            "similarities": similarity.values.numpy(),
        }
