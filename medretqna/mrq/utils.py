import re
from typing import List

from transformers import PreTrainedTokenizer


# TODO: add debug logger maybe
class EntityDecoder:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        tag_sep: str = "-",
        entity_key: str = "entity",
    ):
        """Replacement for huggingface ner pipeline if it doesn't work properly.

        Use to assemble human-readable entities from classified tokens.

        Args:
            tokenizer (transformers.AutoTokenizer): proper transformers tokenizer.
            tag_sep (str, optional): separator in class names for bio tags
                (in B-PERSON "-" is a separator). Defaults to "-".
        """
        self.tokenizer = tokenizer
        self.tag_sep = tag_sep
        self.entity_key = entity_key

    def __call__(self, ents: List[dict]) -> List[dict]:
        """Transform list of token entities from NER model to words.

        Args:
            ents (List[dict]): list of dicts with entities.

        Returns:
            List[dict]: list or assembled entities into words.
        """
        grouped_ents = self._group(ents)
        return self._merge(grouped_ents)

    def _group(self, ents: List[dict]) -> List[List[dict]]:
        """Group each tokens and sub entities to a list of entities

        Each entity in output is represented as a list of tokens/sub-entities.

        Args:
            ents (List[dict]): list of entities.

        Returns:
            List[List[dict]]: grouped into separate lists entities.
        """
        if not ents:
            return []
        res_ents = []

        # TODO: test if it's better to just use another if in the main loop
        for i, ent in enumerate(ents):
            # we are looking for first start token ("B-tag") in the list
            if ent[self.entity_key].startswith(f"B{self.tag_sep}"):
                current_entity = [ent]
                current_entity_class = current_entity[0][self.entity_key].split(
                    self.tag_sep, maxsplit=1
                )[1]
                break
            else:
                continue
        else:
            return []

        # if we have first token, then we can start to search for
        # I tokens with the same tag
        for ent in ents[i + 1 :]:
            # If new B token is found -> it's new entity -> save and start over
            if ent[self.entity_key].startswith(f"B{self.tag_sep}"):
                res_ents.append(current_entity)
                current_entity = [ent]
                current_entity_class = current_entity[0][self.entity_key].split(
                    self.tag_sep, maxsplit=1
                )[1]
            # if it's the same tag with I and it's the next token
            # add to current entity
            elif (
                ent[self.entity_key] == f"I{self.tag_sep}{current_entity_class}"
                and ent["start"] - 1 == current_entity[-1]["end"]
            ):
                current_entity.append(ent)
            # something is broken in the model predictions or predictions are bad
            # so we skip entities that e.g. start with I tag
            else:
                pass

        res_ents.append(current_entity)
        return res_ents

    def _merge(self, grouped_ents: List[List[dict]]) -> List[dict]:
        """Merge tokens and words into single string of the entity

        Args:
            grouped_ents (List[List[dict]]): grouped entities

        Returns:
            List[dict]: final list of entities with correct representation
        """
        res = []
        for ent in grouped_ents:
            temp_ent = {
                "tag": ent[0][self.entity_key].split(self.tag_sep, maxsplit=1)[1],
                "text": self.tokenizer.convert_tokens_to_string([x["word"] for x in ent]),
                "start": ent[0]["start"],
                "end": ent[-1]["end"],
            }
            res.append(temp_ent)
        return res


def fix_sentence(text):
    text = re.sub(r"\s", " ", text).strip()
    if not text.endswith("."):
        text += "."
    if not text[0].isupper():
        text = text[0].upper() + text[1:]
    return text
