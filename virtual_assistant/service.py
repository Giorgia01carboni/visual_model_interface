import torch
import numpy as np
import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

from spatiallm import Layout
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors, cleanup_pcd, Compose
from bbox import BBox3D

# Add parent directory to path to find apply_nms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from apply_nms import apply_nms_to_layout
except ImportError:
    print("Warning: apply_nms.py not found. NMS will be skipped.")
    def apply_nms_to_layout(layout, **kwargs): return layout

from .utils import create_plot 

# --- CONFIGURATION ---
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "code_template.txt")

DETECT_TYPE_PROMPT = {
    "all": "Detect walls, doors, windows, boxes.",
    "arch": "Detect walls, doors, windows.",
    "object": "Detect boxes.",
}

class SpatialService:
    def __init__(self, model_path, device="cuda"):
        print(f"Loading SpatialLM from {model_path}...")
        self.device = device
        
        if os.path.exists(TEMPLATE_PATH):
            with open(TEMPLATE_PATH, "r") as f:
                self.code_template = f.read()
            print(f"✅ Loaded Code Template from {TEMPLATE_PATH}")
        else:
            self.code_template = """
@dataclass
class Bbox:
    class: str
    position_x: int
    position_y: int
    position_z: int
    angle_z: int
    scale_x: int
    scale_y: int
    scale_z: int
"""

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        )
        self.model.to(self.device)
        self.model.set_point_backbone_dtype(torch.float32)
        self.model.eval()
        print("Model Loaded.")
    
    def preprocess(self, pcd_path):
        num_bins = self.model.config.point_config["num_bins"]
        grid_size = Layout.get_grid_size(num_bins)

        pcd = load_o3d_pcd(pcd_path)
        pcd = cleanup_pcd(pcd, voxel_size=grid_size)
        points, colors = get_points_and_colors(pcd)

        transform = Compose([
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=num_bins,
            ),
        ])

        data = transform({"name": "pcd", "coord": points.copy(), "color": colors.copy()})
        pcd_tensor = np.concatenate([data["grid_coord"], data["coord"], data["color"]], axis=1)
        pcd_tensor = torch.as_tensor(np.stack([pcd_tensor], axis=0))
        
        return pcd_tensor, points, colors, np.min(points, axis=0)

    def predict(self, file_path, categories, detect_type="object", density=100000, **kwargs):
        # --- PARAMETERS ---
        t_temp = kwargs.get("temperature", 0.6)
        t_top_k = int(kwargs.get("top_k", 10))
        t_top_p = kwargs.get("top_p", 0.95)
        t_beams = int(kwargs.get("num_beams", 2))
        point_size = kwargs.get("point_size", 2.0)

        # 1. Sanitize Categories
        if categories:
            categories = [c.replace(" ", "_") for c in categories]

        print(f"[SPATIAL-DEBUG] Inference Config: Temp={t_temp}, TopK={t_top_k}, Beams={t_beams}, Cats={categories}")

        # 2. Preprocess
        input_pcd, points, colors, min_extent = self.preprocess(file_path)

        # 3. Prompt Construction
        task_prompt = DETECT_TYPE_PROMPT.get(detect_type, DETECT_TYPE_PROMPT["object"])
        if detect_type == "object" and categories:
            task_prompt = "Detect boxes." 
            task_prompt = task_prompt.replace("boxes", ", ".join(categories))
        elif detect_type != "arch" and categories:
            task_prompt = task_prompt.replace("boxes", ", ".join(categories))
            
        full_prompt = f"<|point_start|><|point_pad|><|point_end|>{task_prompt} The reference code is as followed: {self.code_template}"
        
        # --- RAW TOKENIZATION ---
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)

        # 4. Inference
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids, 
                point_clouds=input_pcd,
                max_new_tokens=2048,
                do_sample=True,
                temperature=t_temp,
                top_p=t_top_p,
                top_k=t_top_k,
                num_beams=t_beams,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode output
        raw_text = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"[SPATIAL-DEBUG] Raw Model Output:\n{raw_text[:200]}...") 

        try:
            # 5. Parse Layout
            layout = Layout(raw_text)
            layout.undiscretize_and_unnormalize(num_bins=self.model.config.point_config["num_bins"])
            layout.translate(min_extent)
            
            # 6. Filter by Category 
            if categories:
                allowed = set(c.lower().strip() for c in categories)
                filtered_bboxes = []
                for b in layout.bboxes:
                    if b.class_name.lower().strip() in allowed:
                        filtered_bboxes.append(b)
                layout.bboxes = filtered_bboxes
            
            # 7. Apply NMS
            layout = apply_nms_to_layout(layout, iou_threshold=0.25, dist_threshold=0.1)
            
            boxes = layout.to_boxes()
            layout_str = layout.to_language_string()
            
            # 8. Create Plot (REMOVED show_boxes argument)
            fig = create_plot(
                points, colors, boxes, 
                max_points=density, 
                point_size=point_size
            )
            
            return {
                "success": True,
                "count": len(boxes),
                "boxes": boxes, 
                "layout_str": layout_str,
                "fig": fig,
                "error": None
            }
            
        except Exception as e:
            return {"success": False, "error": f"Parsing failed: {str(e)}\nRaw Output: {raw_text}"}