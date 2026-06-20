import os
import torch
from .models import load_polyflow_twostage
from stream_manager import SkeletonExtractor, PolylineExtractor, X3DExtractor

class ModelManager:
    """
    Unified manager for loading and sharing PolyFlow models
    across stream processing and video uploads.
    """
    
    BINARY_NAMES = ["Normal", "Anomalous"]
    CLASS_NAMES = [
        "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
        "Fighting", "Normal_Videos", "RoadAccidents", "Robbery",
        "Shooting", "Shoplifting", "Stealing", "Vandalism",
    ]

    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.fusion_model = None
        self.slda = None
        self.rec_model = None
        self.slda_14 = None
        
        self.yolo_model = None
        self.poly_extractor = None
        self.x3d_extractor = None
        
        self.load_errors = []

    def load_all(self, ckpt_dir):
        """Loads the two-stage model pipeline and feature extractors."""
        ckpt_dir = str(ckpt_dir)
        fusion_det_ckpt = os.path.join(ckpt_dir, "PolyGuided_AnomalyDetection_best.pt")
        slda_bin_ckpt   = os.path.join(ckpt_dir, "binary_slda_detection.pth")
        fusion_rec_ckpt = os.path.join(ckpt_dir, "PolyGuided_EventRecognition_best.pt")
        slda_14_ckpt    = os.path.join(ckpt_dir, "slda_14class_recognition.pth")
        yolo_path       = os.path.join(ckpt_dir, "yolo11s-pose.pt")
        
        if not os.path.exists(yolo_path):
            yolo_path = "yolo11s-pose.pt" # Fallback

        try:
            print(f"[ModelManager] Loading two-stage models from {ckpt_dir} ...")
            det_model, slda_bin, slda_14class = load_polyflow_twostage(
                fusion_det_ckpt=fusion_det_ckpt,
                slda_binary_ckpt=slda_bin_ckpt,
                fusion_rec_ckpt=fusion_rec_ckpt if os.path.exists(fusion_rec_ckpt) else None,
                slda_14class_ckpt=slda_14_ckpt if os.path.exists(slda_14_ckpt) else None,
                device=self.device
            )
            self.fusion_model = det_model
            self.slda = slda_bin
            self.slda_14 = slda_14class
            if slda_14class is not None:
                self.rec_model = getattr(slda_14class, "_rec_model", None)
        except Exception as e:
            self.load_errors.append(f"Model stack load error: {e}")
            print(f"[ModelManager] ERROR: {e}")

        try:
            self.yolo_model = SkeletonExtractor(model_path=yolo_path, device=self.device)
            self.poly_extractor = PolylineExtractor()
            self.x3d_extractor = X3DExtractor(device=self.device)
        except Exception as e:
            self.load_errors.append(f"Extractor load error: {e}")
            print(f"[ModelManager] ERROR: {e}")

    def is_ready(self):
        return (self.fusion_model is not None and 
                self.slda is not None and 
                self.yolo_model is not None and 
                self.x3d_extractor is not None)

    def get_status(self):
        return {
            "fusion_det": self.fusion_model is not None,
            "slda_bin": self.slda is not None,
            "fusion_rec": self.rec_model is not None,
            "slda_14": self.slda_14 is not None,
            "yolo": self.yolo_model is not None,
            "x3d": self.x3d_extractor is not None,
            "errors": self.load_errors,
            "device": self.device
        }
