from typing import List
from concurrent.futures import ThreadPoolExecutor
from models import BreastImageModel
from data_processing import Handler
from config import MAX_WORKERS


def apply_pipeline(models: List[BreastImageModel], handler: Handler) -> List[BreastImageModel]:
    funcs = handler.get_augmentations()

    def apply_all(model: BreastImageModel) -> List[BreastImageModel]:
        result = [model]
        for func in funcs:
            next_result = []
            for r in result:
                augmented = func(r)
                next_result.extend(augmented)
            result = next_result
        return result

    all_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        all_augmented = list(executor.map(apply_all, models))

    for variants in all_augmented:
        all_results.extend(variants)

    return all_results
