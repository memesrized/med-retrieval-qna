# Small system for Question Answering with Information Retrieval in Medical domain

## How to run the app

### Params to play with:


1. `medretqna/configs/data_preparation_config.json`
    - "sample_size_per_db": 10
        - There are to datasets with n thousands of examples.
        - Set some big numbers to set the pipeline
        - But be aware that it's time consuming of CPU.
    - "seed": 1337
        - just random seed for reproducibility
    - "batch_len": 5
        - number of records to process at a time
        - bigger number -> faster and more memory consuming
    - "device": "system"
        - cpu / cuda:0 (for gpu) / default "system" will check your system and cuda availability and set proper device
    - "mode": "QA"
        - which data to use from datasets
        - A - answers, Q - questions, QA - concatenated
2. `medretqna/configs/back_config.json`
    - "ner_name": "ukkendane/bert-medical-ner"
        - ner model
    - "emb": "menadsa/S-BioELECTRA"
        - name of embeddings model

Models names you can find in `Models to play` section below.


### Preparation:

To run this app you need to create data (aka database with answers):
1. `pip install -r requirements.txt` (in your venv)
2. `cd medretqna`
3. `python data_preparation.py`

### Launch:

In repository root:

1. `docker-compose up --build`
2. open `http://localhost:8501/` in your browser

## Task description

There is a need to extract the most related to a question records from the knowledge base.

### Input
Input is defined as human speech transformed into text with a black box speech-to-text model.
Text is produced by this model as streamed sequence of sentences.

Possible problems: unclear structure, non-related information, reply time

### Assumptions for simplicity of POC
1. Whole sequence of sentences is received before NLP models turn.
2. Knowledge base is just an array in memory.
3. If conversation is multi-turn all algorithms will be almost the same, but with additional logic:
    - whole conversation can be used
    - only person's speech can be used
    - if there is a need to determine the most recent and valuable replies:
        - for ner and classification tasks more recent classes can replace previous results of models
        - descending weights can be applied to a sequence of text

## Possible solutions

### Named Entity Recognition based

It can be done for general questions, but then it requires too much time, data and resources.

For simplicity let's define the task as "Can I use this medicine in this certain condition?"

With this assumption most cases can be defined as a set of entities:
- person condition
    - use of medicine
    - chronic illness
    - pregnancy
    - etc.
- medicine kind/name
- problem to solve

NER may help with solving of unclear structure and unrelated information as it's extracts only relevant entities.

#### Variations
1. Naive

    In naive approach is simple matching of entities between input and Q&A (already processed by the model) database can be used.

    If should be fast, but a lot of information will be lost due to different forms of words.

2. Naive + static embeddings + ranking by distance

    The same approach but with static embeddings (Word2Vec, fastText, etc.) applied to entities from both sides.

    Embeddings for a text (entities) is just a mean of their vectors.

    Also similarity of texts is cosine or euclid distance between vectors, so they can be sorted (ranked) by this distance.

3. Naive + context embeddings + ranking by distance

    Same, but with context embeddings for entities.

4. NER as classifier + static embeddings + ranking by distance

    NER extracts only related words that it was trained for, but what if information in other words is somewhat related to the situation?

    Input is a sequence of texts, let's say that an element of the sequence is related to the question if there is at least on entity found by NER model.

    Embeddings and ranking strategy is the same as previous variation.

5. NER as classifier + context embeddings + ranking by distance

    Same, but with models as embeddings.

    Context embeddings may be for words (then mean is needed) and for sequences of words. Each of this approaches should be better in terms of scores than classic static embeddings, but slower.

    Embeddings and ranking strategy is the same as previous variation.

#### Pretrained/custom models
There are some NER models and dataasets related to climical/medical/bio/lifescience domain, but it's probably not the best match and there is a need to train or fine-tune on custom data, but of course these pretrained models contain enough information that would be helpful for new dataset creation.

### Classification
#### Variations
1. Classification of whole text

    Answers from Q&A database can be considered as classes for classification.

    Disadvantage of this method is that with each new Answer in database there is a need in new training loop, bacause number of classes is changed.

2. Classification of pieces + embeddings

    Same as "NER as classifier + context/static embeddings + ranking by distance", but with classifier as classifier =)

    Binary classification: related/unrelated

3. Multi-label classification for tags

    Our answers in database can have tags, similar to NER, but tags may not be presented in the text itself.

    With multi-label classifier it's possible to predict `n` classes for a text and then match by these tags with answers in the database.

#### Pretrained/custom models

There are no such pretrained model as for NER, only some base Bio/ClinicalBERTs so custom dataset is necessary.

### Sentence embeddings
The assumption we need for this approach is that even with unrelated information embeddings space will represent the text close to right answer.

Let's take a text that is very related to some records in our database. If we add irrelevant information this text will be less related to those top records, but at the same time it's probably will be less related to other answers from the database. So scores change only in absolute numbers, order of sorted records in database should be the same.

#### Pretrained/custom models

A lot of pretrained embeddings models are available in huggingface, but for better performance it's better to train (or finetune) a model with custom loss that is based on penalty for distance between human questions and answers.

### LLM
LLMs are good for general language understanding due to large amount of training data.

At the same time they are really slow and expensive, so it's good baseline to test other models, but not production solution

#### Pretrained/custom models

It's more about proprietary API vs fine-tuned local model. In case of small amounts of requests proprietary API is much more memory efficient, but then we have problems with legal aspects of data transferring (GDPR, etc.)

## Final architecture

As data is not provided and given time is limited I decided to pick pretrained models and test data from huggingface.

### Data

After my investigation I found these datasets to be the closest to given task open data.

Possible QnA database:

- `medmcqa`
    - QnA dataset with multiple answers
- `medquad` (`AnonymousSub/MedQuAD_47441_Question_Answer_Pairs` on huggingface)
    - QnA dataset

Interesting test set (is not implemented yet):

- `Elfsong/ClinicalDataset`
    - patient-doctor small dialogs and their summaries

### Models and Pipeline

The best solution that is possible to implement without model training (with usage of pretrained models) is NER as classification + sentence embeddings.

There are a lot of general med/bio/clinical models for ner and embeddings, but classification task is too specific to find open-source solution.

LLMs are too big for local experiments and also small LLMs are not that good.

Just embeddings without ner should suffer from irrelevant information without proper fine-tuning. Such tuning can be done but it's time consuming and take a lot of effort.

#### Models to play with

    ner_models = [
        "ukkendane/bert-medical-ner",
        "samrawal/bert-base-uncased_clinical-ner",
        "samrawal/bert-large-uncased_med-ner",
    ]
    emb_models = [
        "emilyalsentzer/Bio_ClinicalBERT",
        "medicalai/ClinicalBERT",
        "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb",
        "menadsa/S-BioELECTRA",
        "TimKond/S-BioLinkBert-MedQuAD",
        "TimKond/S-PubMedBert-MedQuAD",
        "kamalkraj/bioelectra-base-discriminator-pubmed",
    ]

### Retrieval Algorithm

To imitate embeddings search simple matrix multiplication is used to find similarities and then pick sorted top-k results.

In production (e.g. `n` millions of records) it can be implemented via `elastic search` or `faiss` library.

### Backend and UI

Simple `FastAPI` server with hosted models and `Streamlit` for demonstration with nice UI.

Also it's possible to up the whole app with `docker-compose`.
