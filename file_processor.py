from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Tuple, Optional
from io import BytesIO
from PIL import Image
from models import BreastImageModel
import matplotlib.pyplot as plt
from config import DICOM_INFO_PATH, DIRECTORY_PATH, CALC_PATH, MASS_TRAIN_PATH, MASS_TEST_PATH
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.patches as patches

from tqdm import tqdm 
import warnings
warnings.filterwarnings("ignore")


class BreastCancerDataLoader:
    def __init__(self, directory: str, pattern: str = "*.jpg", max_workers: int = 20):
        self.directory = Path(directory)
        self.pattern = pattern
        self.max_workers = max_workers
        self.files = list(self.directory.glob(f"**/{self.pattern}"))

    def read_file(self, path: Path) -> Tuple[str, str]:
        with open(path, 'rb') as f:
            return path.name, f.read()
        
    def get_image_size(self, content: bytes) -> Tuple[int, int]:
        with Image.open(BytesIO(content)) as img:
            return img.width, img.height

    def read_all_files(self) -> List:
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for name, content in executor.map(self.read_file, self.files):
                size = self.get_image_size(content)
                results.append(BreastImageModel(name, content, size))
        return results

    def read_files(self):
        dicom_info = pd.read_csv(DICOM_INFO_PATH)
        image_dir = DIRECTORY_PATH

        # Step 1: Extract and fix paths by series description
        cropped_images = self._get_image_paths_by_description(dicom_info, 'cropped images', image_dir)
        full_mammogram_images = self._get_image_paths_by_description(dicom_info, 'full mammogram images', image_dir)
        ROI_mask_images = self._get_image_paths_by_description(dicom_info, 'ROI mask images', image_dir)

        # Step 2: Build lookup dictionaries keyed by image ID
        full_mammo_dict = self._build_image_dict(full_mammogram_images)
        cropped_images_dict = self._build_image_dict(cropped_images)
        roi_img_dict = self._build_image_dict(ROI_mask_images)

        # Step 3: Load mass train/test CSVs
        mass_train = pd.read_csv(MASS_TRAIN_PATH)
        mass_test = pd.read_csv(MASS_TEST_PATH)

        # Step 4: Fix image paths in train/test data
        self._fix_image_paths_in_data(mass_train, full_mammo_dict, cropped_images_dict, roi_img_dict)
        self._fix_image_paths_in_data(mass_test, full_mammo_dict, cropped_images_dict, roi_img_dict)

        # Step 5: Rename columns and fill missing values
        mass_train = self._prepare_mass_data(mass_train)
        mass_test = self._prepare_mass_data(mass_test)

        # Step 6: Combine datasets
        full_mass = pd.concat([mass_train, mass_test], axis=0)

        # Step 7: Map pathology to labels
        full_mass['labels'] = full_mass['pathology'].replace({'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0})

        # Step 8: For testing, take a subset (replace with full_mass for real use)
        print(len(full_mass))
        copy_mass = full_mass.head(50).copy()

        # Step 9: Load and process images
        copy_mass[['processed_images', 'width', 'height']] = copy_mass['image_file_path'].apply(
                                                                            lambda x: pd.Series(self._load_image(x))
                                                                        )

        # Step 10: Prepare output
        X_resized = copy_mass['processed_images'].tolist()
        num_classes = len(copy_mass['labels'].unique())

        breast_images = [
            BreastImageModel(
                name=Path(row['image_file_path']).name,
                content=row['processed_images'],
                size=(row['width'], row['height']),
                label=row['labels']
            )
            for _, row in copy_mass.iterrows()
            if row['processed_images'] is not None
        ]

        return breast_images, num_classes

    def read_mammograms(self, limit: int = 0):
        dicom_info = pd.read_csv(DICOM_INFO_PATH)
        image_dir = DIRECTORY_PATH

        full_mammogram_images = self._get_image_paths_by_description(dicom_info, 'full mammogram images', image_dir)
        ROI_mask_images = self._get_image_paths_by_description(dicom_info, 'ROI mask images', image_dir)

        full_mammo_dict = self._build_image_dict(full_mammogram_images)
        roi_img_dict = self._build_image_dict(ROI_mask_images)

        mass_train = pd.read_csv(MASS_TRAIN_PATH)
        mass_test = pd.read_csv(MASS_TEST_PATH)

        self._fix_image_paths_in_data(mass_train, full_mammo_dict, {}, roi_img_dict)
        self._fix_image_paths_in_data(mass_test, full_mammo_dict, {}, roi_img_dict)

        mass_data = pd.concat([mass_train, mass_test])
        mass_data = self._prepare_mass_data(mass_data)
        mass_data['labels'] = mass_data['pathology'].replace({'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0})
        
        if limit != 0:
            mass_data = mass_data.head(limit)

        results = []
        results_cropped = []
        for _, row in tqdm(mass_data.iterrows(), total=len(mass_data), desc="Loading mammograms"):
            img = self._load_image(row['image_file_path'])
            if img is None:
                continue

            processed_image, width, height = img
            bbox = self.get_lesion_coords_from_roi_mask(row['ROI_mask_file_path'])

            model = BreastImageModel(
                name=Path(row['image_file_path']).name,
                content=processed_image,
                size=(width, height),
                label=row['labels'],
                bbox=bbox
            )

            results.append(model)

            if bbox:
                x, y, w, h = bbox
                cropped = processed_image[y:y+h, x:x+w]
                cropped_model = BreastImageModel(
                    name=Path(row['image_file_path']).name,
                    content=cropped,
                    size=(w, h),
                    label=row['labels'],
                    bbox=bbox
                )
                results_cropped.append(cropped_model)

        num_classes = len(mass_data['labels'].unique())
        return results, results_cropped, num_classes

    def read_full_mammograms(self, limit: Optional[int] = 50):
        dicom_info = pd.read_csv(DICOM_INFO_PATH)
        image_dir = DIRECTORY_PATH

        full_mammogram_images = self._get_image_paths_by_description(dicom_info, 'full mammogram images', image_dir)
        full_mammo_dict = self._build_image_dict(full_mammogram_images)

        mass_train = pd.read_csv(MASS_TRAIN_PATH)
        mass_test = pd.read_csv(MASS_TEST_PATH)

        self._fix_image_paths_in_data(mass_train, full_mammo_dict, {}, {})
        self._fix_image_paths_in_data(mass_test, full_mammo_dict, {}, {})

        mass_data = pd.concat([mass_train, mass_test])
        mass_data = self._prepare_mass_data(mass_data)
        mass_data['labels'] = mass_data['pathology'].replace({
            'MALIGNANT': 1,
            'BENIGN': 0,
            'BENIGN_WITHOUT_CALLBACK': 0
        })

        if limit:
            mass_data = mass_data.head(limit)

        results = []
        for _, row in tqdm(mass_data.iterrows(), total=len(mass_data), desc="Loading full mammograms"):
            img = self._load_image(row['image_file_path'])
            if img is None:
                continue

            processed_image, width, height = img
            model = BreastImageModel(
                name=Path(row['image_file_path']).name,
                content=processed_image,
                size=(width, height),
                label=row['labels'],
                bbox=None
            )
            results.append(model)

        num_classes = len(mass_data['labels'].unique())
        return results, num_classes

    def read_cropped_mammograms(self, limit: Optional[int] = 50):
        dicom_info = pd.read_csv(DICOM_INFO_PATH)
        image_dir = DIRECTORY_PATH

        cropped_mammogram_images = self._get_image_paths_by_description(dicom_info, 'cropped images', image_dir)

        cropped_mammo_dict = self._build_image_dict(cropped_mammogram_images)

        mass_train = pd.read_csv(MASS_TRAIN_PATH)
        mass_test = pd.read_csv(MASS_TEST_PATH)

        self._fix_image_paths_in_data(mass_train, {}, cropped_mammo_dict, {})
        self._fix_image_paths_in_data(mass_test, {}, cropped_mammo_dict, {})

        mass_data = pd.concat([mass_train, mass_test])
        mass_data = self._prepare_mass_data(mass_data)
        mass_data['labels'] = mass_data['pathology'].replace({'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0})
        
        if limit:
            mass_data = mass_data.head(limit)

        results = []
        for _, row in tqdm(mass_data.iterrows(), total=len(mass_data), desc="Loading mammograms"):
            img = self._load_image(row['cropped_image_file_path'])
            if img is None:
                continue

            processed_image, width, height = img

            model = BreastImageModel(
                name=Path(row['cropped_image_file_path']).name,
                content=processed_image,
                size=(width, height),
                label=row['labels'],
                bbox=None
            )
            results.append(model)

        num_classes = len(mass_data['labels'].unique())
        return results, num_classes

    def get_lesion_coords_from_roi_mask(self, roi_mask_path):
        """Load ROI mask image and return bounding box of lesion in full mammogram coords."""
        absolute_path = os.path.abspath(roi_mask_path)
        mask = cv2.imread(absolute_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)

    def _get_image_paths_by_description(self, dicom_info: pd.DataFrame, description: str, image_dir: str) -> pd.Series:
        """Get image paths filtered by SeriesDescription and fix directory paths."""
        filtered = dicom_info[dicom_info.SeriesDescription == description].image_path
        return filtered.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))

    def _build_image_dict(self, image_paths: pd.Series) -> dict:
        """Build a dictionary mapping image ID (part of the path) to the full path."""
        image_dict = {}
        for path in image_paths:
            key = path.split("/")[6]
            image_dict[key] = path
        return image_dict

    def _fix_image_paths_in_data(self, data: pd.DataFrame, full_mammo_dict: dict, cropped_images_dict: dict, roi_img_dict: dict):
        """Fix image file paths in the dataframe using the lookup dictionaries."""
        for index, row in data.iterrows():
            img_name = row[11].split("/")[2]
            data.at[index, data.columns[11]] = full_mammo_dict.get(img_name, row[11])
            img_name = row[12].split("/")[2]
            data.at[index, data.columns[12]] = cropped_images_dict.get(img_name, row[12])
            img_name = row[13].split("/")[2]
            data.at[index, data.columns[13]] = roi_img_dict.get(img_name, row[13])

    def _prepare_mass_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Rename columns and fill missing values as needed."""
        rename_map = {
            'left or right breast': 'left_or_right_breast',
            'image view': 'image_view',
            'abnormality id': 'abnormality_id',
            'abnormality type': 'abnormality_type',
            'mass shape': 'mass_shape',
            'mass margins': 'mass_margins',
            'image file path': 'image_file_path',
            'cropped image file path': 'cropped_image_file_path',
            'ROI mask file path': 'ROI_mask_file_path',
        }
        data = data.rename(columns=rename_map)
        data['mass_shape'] = data['mass_shape'].bfill()
        data['mass_margins'] = data['mass_margins'].bfill()
        return data

    def _load_image(self, image_path: str):
        """Load an image from an absolute path using OpenCV."""
        absolute_image_path = os.path.abspath(image_path)
        image = cv2.imread(absolute_image_path)
        h, w = image.shape[:2]
        return image, w, h

    def show_images(
        self,
        image_models: List,
        draw_ground_truth: bool = False,
        predicted_bboxes: Optional[List[Tuple[int, int, int, int]]] = None
    ):
        """
        Display images with optional ground truth and predicted bounding boxes.

        Args:
            image_models: List of BreastImageModel instances.
            draw_ground_truth: If True, draw the ground truth bbox (in green).
            predicted_bboxes: Optional list of predicted bounding boxes (x, y, w, h), one per image (in blue).
        """
        num_images = len(image_models)
        rows = (num_images // 3) + (1 if num_images % 3 != 0 else 0)

        fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
        axes = axes.flatten()

        for i, model in enumerate(image_models):
            img = model.content
            ax = axes[i]

            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)

            ax.set_title(f'{model.name}\n{model.size[0]}x{model.size[1]}')
            ax.axis('off')

            # Draw ground truth bbox in green
            if draw_ground_truth and hasattr(model, 'bbox') and model.bbox is not None:
                x, y, w, h = model.bbox
                rect = patches.Rectangle((x, y), w, h, linewidth=1.2, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y - 10, f'GT: {w}x{h}', color='lime', fontsize=6)

            # Draw predicted bbox in blue
            if predicted_bboxes is not None and i < len(predicted_bboxes):
                pred = predicted_bboxes[i]
                if pred is not None:
                    x, y, w, h = pred
                    rect = patches.Rectangle((x, y), w, h, linewidth=1.2, edgecolor='blue', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x, y + h + 10, f'Pred: {w}x{h}', color='blue', fontsize=6)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()