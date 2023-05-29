from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from mrq.logger import get_logger
from mrq.models import EmbedModel
from tqdm.auto import tqdm
from typing_extensions import Self

log = get_logger(__name__)


# TODO: add logger
def load_data(path: str, sample_size: int = 100000, seed: int = 1337) -> pd.DataFrame:
    """Load and preprocess datasets to unify them

    Return pd.DataFrame with cols ["A" (answers), "Q" (questions)] for QnA datasets.

    Args:
        path (str): huggingface dataset name
        sample_size (int, optional): number of records to return. Note that
            it will be min(len(data), sample_size) Defaults to 100000.
        seed (int, optional): random seed. Defaults to 1337.

    Returns:
        pd.DataFrame: pandas dataframe with two columns A and Q
    """
    data = load_dataset(path, split="train").to_pandas()
    # note: datasets for qna sometimes contain a lot of questions
    # but most of the answers may be empty, so we use dropna
    log.info("Loaded data length: {}".format(len(data)))
    data = data.dropna()
    log.info("Data length after dropna: {}".format(len(data)))
    # hardcoded for simplicity, general logic is unnecessary for POC
    if path == "medmcqa":
        ans_dict = {0: "opa", 1: "opb", 2: "opc", 3: "opd"}
        data["Q"] = data["question"]
        data["A"] = data.apply(
            lambda row: row[ans_dict[row["cop"]]] + ". " + row["exp"], axis=1
        )
    elif path == "AnonymousSub/MedQuAD_47441_Question_Answer_Pairs":
        data["Q"] = data["Questions"]
        data["A"] = data["Answers"]
    data = data[["Q", "A"]]
    data = data.sample(min(len(data), sample_size), random_state=seed)
    log.info("Final data length: {}".format(len(data)))
    return data


class AnswerDB:
    def __init__(self, data: List[str]) -> None:
        """Simple mock of database with embeddings.

        Args:
            data (List[str]): _description_
        """
        self.data = np.array(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def encode(
        self,
        emb: EmbedModel,
        tqdm_flag: bool = False,
        batch_len: int = 1,
        final_device="cpu",
    ) -> Self:
        """Encode texts with provided embeddings model

        Args:
            emb (EmbedModel): embeddings model
            tqdm_flag (bool, optional): whether to show tqdm. Defaults to False.
            batch_len (int, optional): number of records to process at a time.
                Larger number -> more memory consumption and less computation time.
                Defaults to 1.
            final_device (str, optional): where to store embeddings after processing

        Raises:
            ValueError: if batch_len is < 1

        Returns:
            Self: class instance itself
        """
        if batch_len < 1:
            raise ValueError("batch_len should be >=1")
        batches = range(0, len(self.data), batch_len)
        batched = (list(self.data[i : i + batch_len]) for i in batches)
        if tqdm_flag:
            proxy = tqdm(batched, total=len(self.data) / batch_len)
        else:
            proxy = batched
        self.embedded = torch.cat([emb(batch) for batch in proxy], dim=0).to(final_device)
        return self


class Query:
    def __init__(self, text: str, embedded: torch.tensor = None):
        """Wrapper class for text query"""
        self.text = text
        self.embedded = embedded

    def embed(self, emb: EmbedModel, final_device="cpu") -> Self:
        self.embedded = emb(self.text).to(final_device)
        return self
