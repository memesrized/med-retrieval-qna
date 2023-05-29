import json
import pickle

import torch
from mrq import PROJECT_PATHS
from mrq.data import AnswerDB, load_data
from mrq.models import EmbedModel

from mrq.logger import get_logger

if __name__ == "__main__":
    log = get_logger(__name__)
    # configs
    with open(PROJECT_PATHS.configs / "back_config.json") as file:
        back_config = json.load(file)

    with open(PROJECT_PATHS.configs / "data_preparation_config.json") as file:
        prep_config = json.load(file)

    # data loading and preparation
    data = load_data(
        "medmcqa", sample_size=prep_config["sample_size"], seed=prep_config["seed"]
    )
    data1 = load_data(
        "AnonymousSub/MedQuAD_47441_Question_Answer_Pairs",
        sample_size=prep_config["sample_size"],
        seed=prep_config["seed"],
    )
    data = data.append(data1)

    # embeddings model loading and data encoding
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    emb_model = EmbedModel(
        back_config["emb"], device=device if prep_config["device"] else "cpu"
    )
    ansdb = AnswerDB(list(set(data["A"]))).encode(
        emb_model, tqdm_flag=True, batch_len=prep_config["batch_len"]
    )
    
    log.info("Ans embedded shape: {}".format(ansdb.embedded.shape))

    # saving data
    with open(back_config["db"], "wb") as file:
        pickle.dump(ansdb, file)
