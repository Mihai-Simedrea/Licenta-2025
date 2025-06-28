import sys
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models import densenet121
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

from architectures.detection_architecture import DenseNetBackbone, BreastCancerDetectionDataset
from config import DIRECTORY_PATH, FILE_PATTERN, MAX_WORKERS
from file_processor import BreastCancerDataLoader
from data_processing import ResizeHandler
from data_pipeline import apply_pipeline
from torchvision.ops import box_iou

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(backbone_type, checkpoint_path):
    """
    Create a Faster R-CNN model with the specified backbone and load weights.
    """
    if backbone_type == "resnet":
        model = fasterrcnn_resnet50_fpn(pretrained=True, num_classes=2)

    elif backbone_type == "densenet":
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
    else:
        raise ValueError("Invalid backbone. Use 'resnet' or 'densenet'.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def evaluate_predictions(targets, predictions, iou_threshold=0.5, score_threshold=0.5):
    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious = []

    for target, pred in zip(targets, predictions):
        gt_boxes = target["boxes"].to(device)
        pred_boxes = pred["boxes"][pred["scores"] >= score_threshold].to(device)

        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue

        ious = box_iou(pred_boxes, gt_boxes)
        matched_gt = set()
        matched_pred = set()

        for pred_idx in range(ious.shape[0]):
            max_iou, gt_idx = ious[pred_idx].max(0)
            if max_iou >= iou_threshold and gt_idx.item() not in matched_gt:
                total_tp += 1
                matched_gt.add(gt_idx.item())
                matched_pred.add(pred_idx)
                all_ious.append(max_iou.item())

        total_fp += len(pred_boxes) - len(matched_pred)
        total_fn += len(gt_boxes) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0

    print("\nEvaluation Results:")
    print(f"True Positives: {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")


def visualize_predictions_vs_ground_truth(images, targets, predictions, score_threshold=0.5):
    for i, (img_tensor, target, prediction) in enumerate(zip(images, targets, predictions)):
        img = TF.to_pil_image(img_tensor.cpu())
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)

        # Ground truth boxes in GREEN
        for box in target["boxes"]:
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)

        # Predictions in RED
        for box, score in zip(prediction["boxes"], prediction["scores"]):
            if score >= score_threshold:
                x1, y1, x2, y2 = box.cpu().numpy()
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor="red", facecolor="none"
                )
                ax.add_patch(rect)

        ax.set_title(f"Image {i}: Green=GT, Red=Prediction")
        plt.axis("off")
        plt.show()


def main():
    if len(sys.argv) != 3:
        print("Usage: python validate.py [resnet|densenet] /path/to/checkpoint.pth")
        sys.exit(1)

    backbone = sys.argv[1]
    checkpoint_path = sys.argv[2]
    print(f"Using backbone: {backbone}")
    print(f"Loading checkpoint from: {checkpoint_path}")

    model = build_model(backbone, checkpoint_path)

    file_processor = BreastCancerDataLoader(DIRECTORY_PATH, FILE_PATTERN, max_workers=MAX_WORKERS)
    images, _, _ = file_processor.read_mammograms()

    classification_chain = ResizeHandler(target_height=224, target_width=224)
    results = apply_pipeline(images, classification_chain)

    dataset = BreastCancerDetectionDataset(results)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x))
    )

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for imgs, targets in tqdm(dataloader):
            imgs = [img.permute(2, 0, 1).to(device) for img in imgs]
            predictions = model(imgs)

            all_targets.extend(targets)
            all_predictions.extend(predictions)

    evaluate_predictions(all_targets, all_predictions)
    visualize_predictions_vs_ground_truth(imgs, targets, predictions)


if __name__ == "__main__":
    main()
