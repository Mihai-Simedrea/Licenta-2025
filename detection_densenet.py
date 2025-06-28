import argparse
import torch
from torchvision.models import densenet121
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as TF
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import DIRECTORY_PATH, FILE_PATTERN, MAX_WORKERS
from file_processor import BreastCancerDataLoader
from data_processing import (
    RotationHandler,
    PaddingHandler,
    ResizeHandler,
    NormalizeHandler,
    BinaryMaskHandler,
)
from data_pipeline import apply_pipeline
from architectures.detection_architecture import DenseNetBackbone, BreastCancerDetectionDataset

HANDLER_MAP = {
    "rotation": RotationHandler,
    "padding": PaddingHandler,
    "resize": lambda: ResizeHandler(target_height=224, target_width=224),
    "normalize": NormalizeHandler,
    "binarymask": BinaryMaskHandler,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN (DenseNet backbone) for Breast Cancer Detection")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--pipeline", nargs="+", default=["resize"], help="Preprocessors to apply")
    parser.add_argument("--checkpoint", type=str, default="detection_densenet.pth", help="Output checkpoint filename")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images")
    return parser.parse_args()

def build_pipeline(handler_names):
    chain = None
    for handler_name in reversed(handler_names):
        handler_cls = HANDLER_MAP.get(handler_name.lower())
        if handler_cls is None:
            raise ValueError(f"Unknown handler: {handler_name}")
        handler_instance = handler_cls() if callable(handler_cls) else handler_cls
        if chain:
            handler_instance.set_next(chain)
        chain = handler_instance
    return chain

def visualize_predictions(images, targets, predictions, score_threshold=0.01):
    for i, (img_tensor, target, prediction) in enumerate(zip(images, targets, predictions)):
        img = TF.to_pil_image(img_tensor.cpu())
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)

        for box in target["boxes"]:
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)

        for box, score in zip(prediction["boxes"], prediction["scores"]):
            if score >= score_threshold:
                x1, y1, x2, y2 = box.cpu().numpy()
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
                ax.add_patch(rect)

        ax.set_title(f"Image {i}: Green=GT, Red=Prediction")
        plt.axis("off")
        plt.show()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    densenet = densenet121(pretrained=True)
    features = densenet.features
    out_channels = 1024
    backbone = DenseNetBackbone(features, out_channels=out_channels)
    backbone.out_channels = out_channels

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator
    )
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    file_processor = BreastCancerDataLoader(DIRECTORY_PATH, FILE_PATTERN, max_workers=MAX_WORKERS)
    images, _, num_classes = file_processor.read_mammograms(limit=args.limit)

    pipeline = build_pipeline(args.pipeline)
    processed_images = apply_pipeline(images, pipeline)

    dataset = BreastCancerDetectionDataset(processed_images)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        for imgs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs = [img.permute(2, 0, 1).to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
        lr_scheduler.step()

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": epoch_loss,
    }, args.checkpoint)
    print(f"Checkpoint saved to {args.checkpoint}")

    model.eval()
    with torch.no_grad():
        imgs_batch, targets_batch = next(iter(dataloader))
        imgs_batch = [img.permute(2, 0, 1).to(device) for img in imgs_batch]
        predictions = model(imgs_batch)
    visualize_predictions(imgs_batch, targets_batch, predictions)

if __name__ == "__main__":
    main()
