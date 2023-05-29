from typing import Union


def output_postprocessing(
    output: dict,
    ner_predictions: Union[dict, list] = None,
    threshold: float = None,
):
    """Additional postprocessing for prediction"""
    # TODO: make it a set?
    ind_to_retain = list(range(len(output["answers"])))
    if threshold:
        ind_to_retain = [i for i, x in enumerate(output["similarities"]) if x > threshold]

    cut_output = {
        k: [el for i, el in enumerate(v) if i in ind_to_retain] for k, v in output.items()
    }

    return {
        "Answers": {
            round(cut_output["similarities"][i], 2): cut_output["answers"][i]
            for i in range(len(cut_output["similarities"]))
        },
        "NER": ner_predictions,
    }
