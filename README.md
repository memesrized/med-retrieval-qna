# Small system for Question Answering with Information Retrieval in Medical domain

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
