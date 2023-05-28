import json
from copy import copy
from pathlib import Path
from typing import List, Union

import numpy as np
from mrq import PROJECT_PATHS
from mrq.logger import get_logger
from mrq.utils import fix_sentence
from segtok.segmenter import split_single
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

log = get_logger(__name__)


class Paraphraser:
    def __init__(
        self,
        s2s_model: str = "Vamsi/T5_Paraphrase_Paws",
        model_max_length: int = 512,
        device="cpu",
    ) -> None:
        """Model for dataset augmentation via paraphrasing.

        Args:
            s2s_model (str, optional): seq-2-seq model from huggingface.
                Defaults to "Vamsi/T5_Paraphrase_Paws".
            model_max_length (int, optional): max context window of the model.
                Defaults to 512.
            device (str, optional): device. Defaults to "cpu".
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(s2s_model).to(device)
        # idk,most of niche models are broken somehow
        # and such ifs are necessary
        if s2s_model == "Vamsi/T5_Paraphrase_Paws":
            s2s_model = "t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(
            s2s_model, model_max_length=model_max_length
        )
        self.device = device

    def __call__(self, text: str, t: float = 1.5, num_seq: int = 5) -> List[str]:
        """_summary_

        Args:
            text (str): input text
            t (float, optional): regulates temperature of the model and
                hence it's creativity. Defaults to 1.5.
            num_seq (int, optional): number of augmented samples to generate
                from single record. Defaults to 5.

        Returns:
            List[str]: paraphrases
        """
        text = "paraphrase: " + text + " </s>"
        outputs = self.model.generate(
            **self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True
            ).to(self.device),
            max_length=256,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=num_seq,
            temperature=t,
        )
        return [
            self.tokenizer.decode(
                x, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for x in outputs
        ]


class Augmentator:
    def __init__(
        self,
        aug_data: Union[str, Path, list] = None,
        aug_num: int = 3,
    ) -> None:
        """Text augmentation with irrelevant information.

        As common human speech contains a lot of unrelated info, let's imitate it.

        Input text is relevant question and output it splitted input text (by sentences)
        which is mixed with sampled augmentations examples.

        Args:
            aug_data (Union[str, Path, list], optional): if provided will be used to
                sample augmentation examples. Defaults to None.
            aug_num (int, optional): number of added sentences for each . Defaults to 3.
        """
        if aug_num < 1:
            raise ValueError("Inappropriate aug_num=={} (expected >=1)".format(aug_num))

        if isinstance(aug_data, str) or isinstance(aug_data, Path):
            with open(aug_data) as file:
                aug_data = json.load(file)
        self.aug_data = aug_data
        self.aug_num = min(aug_num, len(self.aug_data))

    def _sample_aug(self) -> List[str]:
        return list(np.random.choice(self.aug_data, self.aug_num, replace=False))

    def _segment(self, text: str) -> List[str]:
        return split_single(text)

    def _permute(self, seq: list) -> List[str]:
        return list(np.random.permutation(seq))

    def __call__(self, text: str) -> List[str]:
        """Split and augment text"""
        segmented = self._segment(text)
        combined = self._sample_aug() + segmented
        return self._permute(combined)


def augment_init_data(
    inp_file: str,
    out_file: str,
    paraphraser: Paraphraser,
    epochs: int = 1,
    t: float = 1.5,
    num_seq: int = 5,
    return_: bool = False,
):
    """Augment initial data with augmentation by paraphrase.

    For proper comparison of the models we need to augment data with irrelevant info.
    LLMs have failed for such task, so let's generate initial aug examples and
    then get more with T5 paraphrase model.

    Args:
        inp_file (str): file with initial sentences.
        out_file (str): output file with original sentences and paraphrased.
        paraphraser (Paraphraser): paraphrase model.
        epochs (int, optional): number of iterations over sentence. Note, each
            epoch use previous epoch and original sentences. Defaults to 1.
        t (float, optional): regulates temperature of the model and hence
            it's creativity. Defaults to 1.5.
        num_seq (int, optional): number of augmented samples to generate
            from single record. Defaults to 5.
        return_ (bool, optional): whether to return list.
            Defaults to False.

    Returns:
        optional: augmented result
    """
    inp = PROJECT_PATHS.init_data / inp_file
    out = PROJECT_PATHS.data / out_file
    PROJECT_PATHS.data.mkdir(exist_ok=True, parents=True)

    with inp.open() as file:
        examples = json.load(file)
    log.info("Initial number of examples: {}".format(len(examples)))
    log.info(
        "Result number (before set) of examples: {}".format(
            (num_seq**epochs) * len(examples) + len(examples)
        )
    )

    out_examples = copy(examples)
    for ep in range(epochs):
        log.info("Epoch {0} of {1}".format(ep, epochs))
        temp_examples = []
        for ex in tqdm(out_examples):
            temp_examples.extend(paraphraser(ex, t=t, num_seq=num_seq))
        out_examples.extend(temp_examples)

    out_examples = list(set(map(fix_sentence, out_examples)))

    with out.open("w") as file:
        json.dump(out_examples, file, indent=4)

    if return_:
        return out_examples
