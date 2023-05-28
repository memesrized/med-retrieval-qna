from typing import List, Union

import torch
from mrq.utils import EntityDecoder
from transformers import AutoModel, AutoTokenizer, pipeline
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


# TODO: add debug logger maybe
class EmbedModel:
    def __init__(
        self, model_name: str, mode: str = "token", device: str = "cpu"
    ) -> None:
        """Wrapper for embeddings model to make them sentence-wise.

        Ensures to provide normalized embeddings for cosine similarity
        calculation with dot product.

        Args:
            model_name (str): _description_
            mode (str, optional): _description_. Defaults to "token".
            device (str, optional): _description_. Defaults to "cpu".
        """
        self._model = AutoModel.from_pretrained(model_name).to(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.mode = mode
        self._model.to(self.device)

    def __call__(
        self,
        text: Union[str, list],
    ) -> torch.tensor:
        """Embed sentence(s).

        Returns single vector for each input sentence.

        Args:
            text (Union[str, list]): single text or list of texts.

        Returns:
            torch.tensor: tensor [N,M] where N - number of inputs and
                M - hidden state size of the model
        """
        tokenized = self._tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            emb = self._model(**tokenized)
        pooled = self._pool(
            emb, tokenized["attention_mask"] if self.mode == "token" else None
        )
        return self._norm(pooled)

    def _norm(self, pooled: torch.tensor) -> torch.tensor:
        """Apply normalization to a tensor by rows"""
        return pooled / torch.linalg.norm(pooled, dim=1, keepdim=True)

    def _pool(
        self,
        emb: BaseModelOutputWithPoolingAndCrossAttentions,
        mask: torch.tensor = None,
    ) -> torch.tensor:
        """To get a single

        Args:
            emb (BaseModelOutputWithPoolingAndCrossAttentions): output of the model.
            mask (torch.tensor, optional): attention mask, needed to exclude padding tokens
                from calculation of the mean. Defaults to None.

        Returns:
            torch.tensor: single vector for each sentence
        """
        if self.mode == "token":
            masked = emb["last_hidden_state"] * mask.unsqueeze(-1)
            return masked.sum(axis=1) / mask.sum(-1, keepdim=True)
        elif self.mode == "pooler":
            return emb["pooler_output"]
        else:
            raise ValueError("Unknown mode: {}".format(self.mode))


class NERModel:
    def __init__(
        self, model_name: str, agg: str = "first", device: str = "cpu"
    ) -> None:
        """Wrapped for huggingface NER pipeline

        This wrapper is needed to fix some models issues.

        Args:
            model_name (str): huggingface model name
            agg (str, optional): aggregation stategy for ner pipeline.
                Defaults to "first". Use "first" or "simple".
            device (str, optional): device to use. Defaults to "cpu".
        """
        self._model = pipeline(
            task="ner", model=model_name, aggregation_strategy=agg, device=device
        )
        # hardcoded due to limited number of models and POC format of the task
        if model_name == "ukkendane/bert-medical-ner":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._decoder = EntityDecoder(
                tokenizer=tokenizer, entity_key="entity_group", tag_sep="_"
            )
        else:
            self._decoder = self._default_decoder

    def __call__(
        self, text: Union[str, List[str]]
    ) -> Union[List[dict], List[List[dict]]]:
        """Ner prediction method

        Args:
            text (str): input text or list of texts

        Returns:
            List[dict]: list of entities or list of lists of entities
        """
        model_result = self._model(text)
        if isinstance(text, list):
            decoded = [self._decoder(x) for x in model_result]
        else:
            decoded = self._decoder(model_result)
        return decoded

    def _default_decoder(self, ents: List[dict]) -> List[dict]:
        return [
            {
                "tag": ent["entity_group"],
                "text": ent["word"],
                "start": ent["start"],
                "end": ent["end"],
            }
            for ent in ents
        ]

class NERClassifier(NERModel):
    """Classifier upon NER model"""

    def __call__(self, text: Union[List[str], str]) -> List[bool]:
        """Classify input text

        Args:
            text (Union[List[str], str]): input text or list of texts.

        Returns:
            List[bool]: classes (1 - medical relate, 0 - unrelated)
        """
        if isinstance(text, str):
            text = [text]
        res = super().__call__(text)
        return list(map(bool, res))

    def extract(self, text: Union[List[str], str]) -> List[str]:
        """Classify and return only related records.

        Args:
            text (Union[List[str], str]): input text or list of texts.

        Returns:
            List[str]: list of related records.
        """
        if isinstance(text, str):
            text = [text]
        return [t for t, flag in zip(text, self(text)) if flag]
