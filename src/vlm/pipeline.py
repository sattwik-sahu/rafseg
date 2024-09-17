import numpy as np
import torch
from PIL.Image import Image, open as open_image
from utils.vision.embeddings.image_vector_store import (
    ImageVectorStore,
    load_image_vector_store,
    PromptImageDocument
)
from utils.vision.embeddings.imgbeddings_pipeline import (
    ImgbeddingsPipeline,
    Imgbeddings,
)
from utils.vision.seg_gpt.helpers import load_model
from utils.vision.seg_gpt.model import SegGPT, seg_gpt_pretrained_model_builder
from utils.vision.seg_gpt import SegGPT_Inference
from pathlib import Path
import typing as t


class Pipeline:
    def __init__(
        self,
        vector_store_path: str | Path,
        seg_gpt_weights_path: str | Path,
    ) -> None:
        seg_gpt: SegGPT = load_model(
            weights_path=seg_gpt_weights_path,
            model_builder=seg_gpt_pretrained_model_builder,
            seg_type="instance",
            device="cuda",
        ).to("cuda")
        self.inference: SegGPT_Inference = SegGPT_Inference(
            model=seg_gpt, device="cuda"
        )
        self.embedding_pipeline = ImgbeddingsPipeline(model=Imgbeddings(gpu=True))
        self.vector_store: ImageVectorStore = load_image_vector_store(
            path=vector_store_path
        )

    def run(self, x: Image, k: int = 3) -> t.Tuple[t.List[PromptImageDocument], torch.Tensor]:
        """
        Runs the pipeline on the given input `x`.

        Args:
            x (Image): The image input into the pipeline.
            k (int): How many of the top matches should be used as
                prompt images?

        Returns:
            t.Tuple[t.List[PromptImageDocument], torch.Tensor]:
                A tuple containing the best matches and the output tensor.
        """
        # Create embedding from query image
        query_embedding = self.embedding_pipeline(x=x)

        # Get best matches from iamge vector store
        best_matches: t.List[PromptImageDocument] = self.vector_store.retrieve(query_embedding=query_embedding, k=k)

        # Get the images and masks from the best matches
        prompt_images: t.List[Image] = [doc.image or open_image(doc.image_path) for doc in best_matches]
        prompt_masks: t.List[Image] = [doc.mask or open_image(doc.mask_path) for doc in best_matches]
        
        # Perform few shot segmentation on the query_image
        # using prompt_images and prompt_masks
        output: torch.Tensor = self.inference(
            query_image=x,
            prompt_images=prompt_images,
            prompt_masks=prompt_masks
        )

        return best_matches, output
