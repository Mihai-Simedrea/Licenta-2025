import argparse
import torch
from torchvision.models import densenet121
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from architectures.detection_architecture import DenseNetBackbone
from torchvision.models.detection import fasterrcnn_resnet50_fpn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Run detection on an image using a trained model")
    parser.add_argument("--backbone", type=str, required=True, help="resenet or densenet")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.pth)")
    parser.add_argument("--score_threshold", type=float, default=0.5, help="Minimum score threshold for predictions")
    return parser.parse_args()

def build_model(backbone_type, checkpoint_path):
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

def preprocess_image(image_path, device):
    img = Image.open(image_path).convert("RGB")
    img_tensor = TF.to_tensor(img)
    img_tensor = TF.resize(img_tensor, [224, 224])
    return img_tensor.to(device)

def visualize_predictions(image_tensor, predictions, score_threshold=0.5):
    img = TF.to_pil_image(image_tensor.cpu())
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)

    for box, score in zip(predictions["boxes"], predictions["scores"]):
        if score >= score_threshold:
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x1, y1, f"{score:.2f}", color="yellow", fontsize=12, backgroundcolor="black")

    ax.set_title("Detected Boxes")
    plt.axis("off")
    plt.show()

def main():
    args = parse_args()

    print("[INFO] Building model...")
    model = build_model(args.backbone, args.model_path)

    print("[INFO] Loading and preprocessing image...")
    img_tensor = preprocess_image(args.image_path, device)

    print("[INFO] Running detection...")
    with torch.no_grad():
        predictions = model([img_tensor])[0]

    print("[INFO] Visualizing predictions...")
    visualize_predictions(img_tensor, predictions, score_threshold=args.score_threshold)

if __name__ == "__main__":
    main()
