import json

from app_utils.load import load_on_startup
from app_utils.processing import output_postprocessing
from app_utils.schemes import UserQuery
from fastapi import FastAPI
from mrq.data import Query
from mrq.logger import get_logger
from segtok.segmenter import split_single

with open("configs/back_config.json") as file:
    config = json.load(file)

log = get_logger(__name__)

app = FastAPI()

ner, emb, db, retriever = load_on_startup(config)


@app.post("/query")
async def query_base(q: UserQuery):
    ner_predictions = ner.extract(split_single(q.text))
    if q.return_ner:
        ner_entities = ner.ner(split_single(q.text))
    else:
        ner_entities = ner_predictions
    text = " ".join(ner_predictions)
    query = Query(text).embed(emb)
    res = retriever.find_it(query, top_k=q.topk)
    log.debug("res: {}".format(res))
    return output_postprocessing(res, ner_predictions=ner_entities, threshold=q.threshold)
