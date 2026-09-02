"""
Multi-Photon Microscopy (MPM) and Pathology Dataset Loader - v6.
Features:
  - Oversampling of minority classes (DCIS and IDC duplicated to match DCISM 7 cases)
  - Heavy chromatic & spatial data augmentations for minority class TIF tiles
  - MONAI / PyTorch torchvision augmentation pipeline
"""

import os
import glob
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import List, Tuple, Dict, Any, Optional
import torchvision.transforms as T


class MPMSequenceDataset(Dataset):
    """
    Dataset PyTorch para secuencias de imagenes de Microscopia Multi-Fotones (MPM TIF).
    Soporta oversampling de clases minoritarias (DCIS e IDC) y aumentos cromaticos pesados.
    """
    
    def __init__(
        self,
        dataset_root: str = r"C:\Users\loapi\Downloads\PKG - HE-vs-MPM",
        image_size: Tuple[int, int] = (224, 224),
        max_seq_len: int = 10,
        is_training: bool = True,
        oversample_minority: bool = True
    ):
        self.dataset_root = dataset_root
        self.mpm_dir = os.path.join(dataset_root, "MPM image")
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.is_training = is_training
        
        self.cases_data = self._index_mpm_cases()
        self.raw_case_ids = sorted(list(self.cases_data.keys()))
        
        self.label_map = {"DCIS": 0, "DCISM": 1, "IDC": 2}
        
        # Balancear dataset por oversampling en entrenamiento
        if is_training and oversample_minority:
            self.case_ids = self._build_balanced_case_list()
        else:
            self.case_ids = self.raw_case_ids

        # Augmentation heavy pipeline for training
        self.heavy_transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=90),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.eval_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _index_mpm_cases(self) -> Dict[str, Dict[str, Any]]:
        cases = {}
        if not os.path.exists(self.mpm_dir):
            return cases
            
        tif_files = sorted(glob.glob(os.path.join(self.mpm_dir, "*.tif")))
        diag_map = {11: "DCIS", 12: "IDC"}
        
        for filepath in tif_files:
            filename = os.path.basename(filepath)
            match = re.match(r"Case(\d+)-ROI-(\d+)\.tif", filename)
            if match:
                case_num = int(match.group(1))
                roi_num = int(match.group(2))
                diagnosis = diag_map.get(case_num, "DCISM")
                case_key = f"Case{case_num}-{diagnosis}"
                
                if case_key not in cases:
                    cases[case_key] = {
                        "case_number": case_num,
                        "diagnosis": diagnosis,
                        "tif_paths": []
                    }
                cases[case_key]["tif_paths"].append((roi_num, filepath))
                
        for c in cases:
            cases[c]["tif_paths"].sort(key=lambda x: x[0])
            cases[c]["tif_paths"] = [p[1] for p in cases[c]["tif_paths"]]
            
        return cases

    def _build_balanced_case_list(self) -> List[str]:
        """Duplica los casos minoritarios (DCIS e IDC) para igualar los 7 casos de DCISM."""
        dcism_cases = [c for c in self.raw_case_ids if "DCISM" in c]
        dcis_cases  = [c for c in self.raw_case_ids if "DCIS" in c and "DCISM" not in c]
        idc_cases   = [c for c in self.raw_case_ids if "IDC" in c]
        
        target_count = len(dcism_cases) # 7 casos
        
        balanced = list(dcism_cases)
        if dcis_cases:
            balanced.extend(dcis_cases * (target_count // len(dcis_cases)))
        if idc_cases:
            balanced.extend(idc_cases * (target_count // len(idc_cases)))
            
        np.random.shuffle(balanced)
        return balanced

    def _load_and_preprocess_image(self, img_path: str, is_minority: bool = False) -> torch.Tensor:
        try:
            pil_img = Image.open(img_path).convert("RGB")
            
            if self.is_training and is_minority:
                return self.heavy_transform(pil_img)
            elif self.is_training:
                return self.heavy_transform(pil_img)
            else:
                return self.eval_transform(pil_img)
        except Exception:
            return torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_id = self.case_ids[idx]
        case_info = self.cases_data[case_id]
        
        tif_paths = case_info["tif_paths"]
        diagnosis = case_info["diagnosis"]
        label = self.label_map.get(diagnosis, 1)
        is_minority = (diagnosis in ["DCIS", "IDC"])
        
        seq_tensors = []
        for path in tif_paths[:self.max_seq_len]:
            t = self._load_and_preprocess_image(path, is_minority=is_minority)
            seq_tensors.append(t)
            
        actual_seq_len = len(seq_tensors)
        while len(seq_tensors) < self.max_seq_len:
            seq_tensors.append(torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32))
            
        sequence_tensor = torch.stack(seq_tensors, dim=0)
        
        return {
            "case_id": case_id,
            "sequence_tensor": sequence_tensor,
            "seq_len": actual_seq_len,
            "label": torch.tensor(label, dtype=torch.long),
            "diagnosis": diagnosis
        }


def get_mpm_dataloader(
    dataset_root: str = r"C:\Users\loapi\Downloads\PKG - HE-vs-MPM",
    batch_size: int = 3,
    shuffle: bool = True,
    oversample_minority: bool = True
) -> DataLoader:
    dataset = MPMSequenceDataset(dataset_root=dataset_root, is_training=shuffle, oversample_minority=oversample_minority)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
