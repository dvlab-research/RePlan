import argparse
import os
import sys
import json
import base64
import threading
import re
from io import BytesIO
from pathlib import Path
from datetime import datetime
import torch
import gradio as gr
import numpy as np
from PIL import Image, ImageOps

# ================= Dependency Check =================
try:
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import JSONResponse, HTMLResponse
except ImportError:
    print("❌ Error: necessary dependencies missing, please run: pip install fastapi python-multipart uvicorn")
    sys.exit(1)

current_path = Path(__file__).resolve()
project_root = str(current_path.parents[2]) if len(current_path.parents) > 2 else str(current_path.parent)
if project_root not in sys.path: sys.path.insert(0, project_root)

gpu_lock = threading.Lock()
PROGRESS_STATES = {}

# ================= Pipeline Setup =================
# Mock Pipeline (when real environment is missing)
class MockPipeline:
    def __init__(self, *args, **kwargs):
        print("\n[Warning] Using Mock mode")
    def get_vlm_response(self, instruction, image_path):
        return '<gen_image>remove the object</gen_image><region>[{"bbox_2d":[100,100,300,300],"hint":"target_object"}]</region>', []
    def region_edit_with_attention(self, image, instruction, response, output_dir, **kwargs):
        img = Image.open(image).convert("RGB")
        os.makedirs(output_dir, exist_ok=True)
        # Mock processing
        return ImageOps.invert(img), "mock", {"output_dir": output_dir}

try:
    from replan.pipelines.replan import RePlanPipeline
    print("✅ Successfully imported RePlanPipeline")
except ImportError:
    print("⚠️ replan library not found, switching to MockPipeline mode")
    RePlanPipeline = MockPipeline

def image_to_base64(image):
    if image is None: return None
    try:
        if isinstance(image, str): image = Image.open(image)
        if isinstance(image, np.ndarray): image = Image.fromarray(image)
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    except Exception as e:
        print(f"Base64 error: {e}")
        return None

def _parse_torch_dtype(dtype_str: str) -> torch.dtype:
    s = (dtype_str or "").lower().strip()
    if s in {"bf16", "bfloat16"}: return torch.bfloat16
    if s in {"fp16", "float16", "half"}: return torch.float16
    return torch.float32

# ================= Helper Functions =================
def expand_key_to_split(key):
    if key == 'Background':
        return ['Noise Background', 'Image Background']
    bbox_match = re.match(r'^BBox (\d+)$', key)
    if bbox_match:
        idx = bbox_match.group(1)
        return [f'Noise BBox {idx}', f'Image BBox {idx}']
    return [key]

def generate_default_split_rules(num_regions, has_image_prompt=True):
    # 1. logical components
    logical_components = ['Main Prompt']
    if has_image_prompt: 
        logical_components.append('Image Prompt')
    
    hint_components = [f'Hint {i+1}' for i in range(num_regions)]
    bbox_components = [f'BBox {i+1}' for i in range(num_regions)]
    
    logical_components.extend(hint_components)
    logical_components.extend(bbox_components)
    logical_components.append('Background')
    
    # 2. logic rules
    logical_rules = { (q, k): False for q in logical_components for k in logical_components }

    # (A) Self Attention
    for c in logical_components:
        logical_rules[(c, c)] = True

    # (B) Image Prompt Logic 
    if has_image_prompt:
        for c in logical_components:
            if c != 'Image Prompt':
                logical_rules[('Image Prompt', c)] = True
                logical_rules[(c, 'Image Prompt')] = True
    
    # (C) Image Internal & Cross
    img_logical_group = bbox_components + ['Background']
    
    for q in img_logical_group:
        for k in img_logical_group:
            logical_rules[(q, k)] = True
        # Cross with Main Prompt
        logical_rules[(q, 'Main Prompt')] = True
        logical_rules[('Main Prompt', q)] = True

    # (D) Hint <-> BBox
    for i in range(num_regions):
        bbox = f'BBox {i+1}'
        hint = f'Hint {i+1}'
        logical_rules[(bbox, hint)] = True
        # Hint sees all image parts
        for img_c in img_logical_group:
            logical_rules[(hint, img_c)] = True

    # 3. Split
    final_rules = {}
    for (q_logic, k_logic), val in logical_rules.items():
        if val:
            for q in expand_key_to_split(q_logic):
                for k in expand_key_to_split(k_logic):
                    final_rules[(q, k)] = True
                    
    all_real = []
    for lc in logical_components: all_real.extend(expand_key_to_split(lc))
    for c in all_real: final_rules[(c, c)] = True

    return final_rules

# ================= Frontend Template =================

ICONS = {
    "layer": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>',
    "image": '<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>',
    "trash": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>'
}

FULL_PAGE_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RePlan Pro</title>
    <style>
        body { margin: 0; padding: 0; background: #f3f4f6; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; height: 100vh; overflow: hidden; display: flex; flex-direction: column; }
        .header { height: 50px; background: white; border-bottom: 1px solid #e5e7eb; display: flex; align-items: center; padding: 0 20px; flex-shrink: 0; }
        .header h1 { font-size: 18px; color: #4f46e5; margin: 0; font-weight: 700; }
        
        .workspace { display: flex; flex: 1; overflow: hidden; gap: 10px; padding: 10px; }
        .col { background: white; border: 1px solid #e5e7eb; border-radius: 8px; display: flex; flex-direction: column; padding: 15px; gap: 10px; overflow-y: auto; }
        
        /* === Modified Column Layout === */
        .col-left { width: 300px; flex-shrink: 0; }
        .col-center { flex: 1.5; min-width: 0; padding: 0 !important; border: none; background: transparent; overflow: hidden; }
        
        /* Right is flexible (flex 1) but has min/max constraints */
        .col-right { 
            flex: 1; 
            min-width: 350px; 
            max-width: 600px; 
            flex-shrink: 0; 
        }
        
        .label { font-size: 11px; font-weight: 700; color: #6b7280; text-transform: uppercase; margin-bottom: 4px; display: block; }
        
        .btn { width: 100%; padding: 10px; border-radius: 6px; font-weight: 600; cursor: pointer; border: none; color: white; transition: 0.2s; }
        .btn:hover { opacity: 0.9; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; background: #9ca3af !important; }
        .btn-primary { background: #4f46e5; }
        .btn-success { background: #10b981; }
        
        .panel-input { width: 100%; border: 1px solid #d1d5db; border-radius: 6px; padding: 8px; font-size: 13px; margin-bottom: 8px; font-family: inherit; box-sizing: border-box; }
        .input-box { width: 100%; padding: 8px; border: 1px solid #d1d5db; border-radius: 6px; resize: vertical; box-sizing: border-box; font-family: inherit; }
        .input-box:focus, .panel-input:focus { outline: none; border-color: #4f46e5; }

        input[type=range] { -webkit-appearance: none; width: 100%; background: transparent; }
        input[type=range]:focus { outline: none; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; height: 16px; width: 16px; border-radius: 50%; background: #4f46e5; cursor: pointer; margin-top: -6px; }
        input[type=range]::-webkit-slider-runnable-track { width: 100%; height: 4px; cursor: pointer; background: #e5e7eb; border-radius: 2px; }
        
        .upload-area { border: 2px dashed #d1d5db; border-radius: 6px; padding: 20px; text-align: center; cursor: pointer; transition: 0.2s; position: relative; background: #f9fafb; display: block; }
        .upload-area:hover { border-color: #4f46e5; background: #eff6ff; }
        .preview-img { max-width: 100%; max-height: 150px; margin-top: 10px; display: none; border-radius: 4px; object-fit: contain; margin: 10px auto 0 auto; }
        
        /* === Modified Result Container === */
        .result-container { 
            flex: 1; 
            background: #1f2937; 
            border-radius: 6px; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            position: relative; 
            min-height: 400px; 
            overflow: hidden; 
        }
        
        .result-img { max-width: 100%; max-height: 100%; object-fit: contain; display: none; }
        .loading-overlay { position: absolute; inset: 0; background: rgba(0,0,0,0.6); color: white; display: none; align-items: center; justify-content: center; font-weight: bold; flex-direction: column; gap: 10px; z-index: 50; }
        
        .ex-list { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 5px; }
        .ex-item { aspect-ratio: 1; border-radius: 6px; cursor: pointer; border: 2px solid transparent; overflow: hidden; position: relative; background: #eee; }
        .ex-item:hover { border-color: #4f46e5; }
        .ex-item img { width: 100%; height: 100%; object-fit: cover; }
        .ex-item .ex-desc { display: none; position: absolute; bottom:0; left:0; right:0; background:rgba(0,0,0,0.7); color:#fff; font-size:9px; padding:2px; text-align:center; pointer-events:none; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .ex-item:hover .ex-desc { display: block; }

        /* --- Attention Matrix Styles (Updated for Collapse & Fix) --- */
        .attn-container { border: 1px solid #374151; border-radius: 6px; background: #111827; overflow: hidden; margin-top: 5px; flex-shrink: 0; }
        .attn-header { background: #1f2937; padding: 6px 10px; font-size: 11px; color: #9ca3af; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #374151; cursor: pointer; user-select: none; min-height: 32px; }
        .attn-header:hover { background: #374151; color: white; }
        
        /* Collapsible States */
        .attn-table-wrap { overflow: auto; max-height: 300px; min-height: 120px; padding: 0; transition: all 0.2s ease; } 
        .attn-table-wrap.collapsed { max-height: 0; min-height: 0; opacity: 0; pointer-events: none; }
        
        .attn-caret { display: inline-block; transition: transform 0.2s; margin-right: 6px; font-size: 10px; }
        .attn-caret.closed { transform: rotate(-90deg); }

        .attn-table { border-collapse: separate; border-spacing: 0; width: max-content; font-family: ui-monospace, monospace; font-size: 10px; }
        .attn-table th { color: #9ca3af; padding: 4px; font-weight: normal; border-bottom: 1px solid #374151; border-right: 1px solid #374151; min-width: 40px; text-align: center; background: #1f2937; position: sticky; top: 0; z-index: 10; }
        .attn-table td { border-bottom: 1px solid #374151; border-right: 1px solid #374151; padding: 0; text-align: center; height: 24px; vertical-align: middle; }
        .attn-table .row-label { text-align: left; padding-left: 8px; color: #d1d5db; background: #1f2937; min-width: 100px; position: sticky; left: 0; z-index: 5; border-right: 2px solid #374151; }
        .attn-cell { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; cursor: pointer; min-width: 24px; }
        .attn-cell:hover { background: rgba(255,255,255,0.1); }
        .attn-v { color: #10b981; font-weight: bold; font-size: 12px; }
        .attn-x { color: #ef4444; font-weight: bold; font-size: 12px; opacity: 0.3; }

        /* === Editor Styles === */
        .editor-container * { box-sizing: border-box; }
        .editor-container { height: 100%; width: 100%; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; background-color: white; }
        .toolbar { background-color: #f9fafb; border-bottom: 1px solid #e5e7eb; padding: 0.5rem 1rem; height: 3rem; display: flex; align-items: center; flex-shrink: 0; }
        .toolbar-title { display: flex; align-items: center; gap: 0.5rem; color: #374151; font-weight: 600; font-size: 0.875rem; }
        
        .main-layout { display: flex; flex: 1; overflow: hidden; flex-direction: column; }
        .canvas-section { flex: 1; background-color: #1f2937; position: relative; display: flex; align-items: center; justify-content: center; overflow: hidden; width: 100%; }
        
        .data-section { flex: 0 0 220px; background-color: white; border-top: 1px solid #e5e7eb; display: flex; flex-direction: row; width: 100%; z-index: 20; }
        .panel-section, .list-section, .selection-panel { flex: 1; border-right: 1px solid #f3f4f6; min-width: 0; padding: 1rem; display: flex; flex-direction: column; }
        .selection-panel { border-right: none; background-color: #fcfcfc; }
        .selection-panel.disabled { opacity: 0.5; pointer-events: none; }

        .empty-state { text-align: center; color: #6b7280; }
        .editor-stage { position: relative; line-height: 0; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3); }
        .target-img { display: block; max-width: 100%; max-height: 100%; width: auto; height: auto; object-fit: contain; user-select: none; }
        .overlay { position: absolute; inset: 0; width: 100%; height: 100%; z-index: 10; cursor: crosshair; }
        .hidden { display: none !important; }
        
        .bbox { position: absolute; border: 2px solid; cursor: move; box-sizing: border-box; }
        .bbox.selected { border: 2px solid #fff !important; box-shadow: 0 0 0 1px rgba(0,0,0,0.5); z-index: 100; }
        
        .bbox-label { 
            position: absolute; 
            top: -24px; left: -2px; 
            padding: 4px 8px; 
            font-size: 12px; color: white; font-weight: 600; 
            border-radius: 4px 4px 0 0;
            pointer-events: none; 
            white-space: nowrap; 
            max-width: 200px; 
            overflow: hidden; 
            text-overflow: ellipsis;
            z-index: 200; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
            line-height: 1.2;
        }
        
        .resize-handle { position: absolute; width: 10px; height: 10px; background-color: white; border: 1px solid #333; display: none; z-index: 101; }
        .bbox.selected .resize-handle { display: block; }
        .handle-nw { top: -5px; left: -5px; cursor: nw-resize; }
        .handle-ne { top: -5px; right: -5px; cursor: ne-resize; }
        .handle-sw { bottom: -5px; left: -5px; cursor: sw-resize; }
        .handle-se { bottom: -5px; right: -5px; cursor: se-resize; }

        .list-item { padding: 5px; cursor: pointer; display: flex; align-items: center; gap: 5px; border-radius: 4px; margin-bottom: 2px; }
        .list-item:hover { background-color: #f3f4f6; }
        .list-item.active { background-color: #eff6ff; border-left: 3px solid #3b82f6; }
        .color-dot { width: 12px; height: 12px; border-radius: 50%; }
        .item-label { font-size: 12px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        
        .color-row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 5px; }
        .color-opt { width: 20px; height: 20px; border-radius: 50%; cursor: pointer; border: 2px solid transparent; transition: transform 0.1s; }
        .color-opt:hover { transform: scale(1.1); border-color: #999; }
        
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>

    <div class="header">
        <h1>🎨 RePlan Pro <span style="font-size:12px; font-weight:normal; color:#6b7280; margin-left:10px;">Custom Interface</span></h1>
    </div>

    <div class="workspace">
        <div class="col col-left">
            <div class="label">1. Upload Image</div>
            <label class="upload-area" id="upload-trigger">
                <span id="upload-text">Click to Upload</span>
                <input type="file" id="file-input" accept="image/*" style="display:none" onchange="window.App.loadFile(this.files[0])">
                <img id="input-preview" class="preview-img">
            </label>
            
            <div class="label" style="margin-top:10px;">Instruction</div>
            <textarea id="instruction" class="input-box" rows="3" placeholder="e.g. Remove the object..."></textarea>
            
            <button id="btn-plan" class="btn btn-primary" onclick="window.App.runPlan()">✨ Generate Plan</button>
            
            <div style="margin-top:20px; border-top:1px solid #e5e7eb; padding-top:10px;">
                <div class="label">Examples</div>
                <div class="ex-list" id="example-list"></div>
            </div>
        </div>

        <div class="col col-center">
            <div class="editor-container" id="editor-root">
                <div class="toolbar">
                    <div class="toolbar-title">
                        <span style="color:#4f46e5; display:flex; margin-right:5px;">[[ICON_LAYER]]</span>
                        <span>Editor Workspace</span>
                    </div>
                </div>
                <div class="main-layout">
                    <div class="canvas-section" id="workspace">
                        <div id="empty-state" class="empty-state">
                            <div class="empty-icon">[[ICON_IMAGE]]</div>
                            <p class="empty-text">Waiting for Image...</p>
                        </div>
                        <div id="editor-stage" class="editor-stage hidden">
                            <img id="target-img" src="" class="target-img" draggable="false">
                            <div id="overlay" class="overlay"></div>
                        </div>
                    </div>
                    <div class="data-section">
                        <div class="panel-section" style="background-color: #f9fafb;">
                            <span class="label">Global Prompt</span>
                            <textarea id="global-prompt" class="panel-input" style="height:100%; resize:none;" placeholder="Global editing prompt..."></textarea>
                        </div>
                        
                        <div class="list-section">
                            <div class="label" style="display:flex; justify-content:space-between">
                                <span>Regions</span>
                                <span id="region-count" style="background:#f3f4f6; padding:0 6px; border-radius:10px; font-size:10px;">0</span>
                            </div>
                            <div id="region-list" style="flex:1; overflow-y:auto;"></div>
                        </div>

                        <div id="selection-panel" class="selection-panel disabled">
                            <div class="label" style="display:flex; justify-content:space-between; align-items:center;">
                                <span style="color:#4f46e5;">Selected Region</span>
                                <button onclick="window.EditorApp.deleteSelected()" style="background:none; border:none; cursor:pointer; color:#ef4444;" title="Delete">[[ICON_TRASH]]</button>
                            </div>
                            <textarea id="label-input" class="panel-input" rows="4" style="resize:vertical; min-height:80px;" placeholder="Hint Label (Enter to describe)..."></textarea>
                            
                            <div class="label">Color</div>
                            <div id="color-row" class="color-row"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="col col-right">
            <div class="label">Result</div>
            <div class="result-container">
                <img id="result-img" class="result-img">
                <div id="result-placeholder" style="color:#9ca3af; font-size:14px;">Result will appear here</div>
                <div id="loading-mask" class="loading-overlay">
                    <div style="width:24px; height:24px; border:3px solid #fff; border-top-color:transparent; border-radius:50%; animation: spin 1s linear infinite;"></div>
                    <div id="loading-text">Processing...</div>
                    <div id="progress-bar-container" style="width:80%; height:4px; background:rgba(255,255,255,0.3); border-radius:2px; margin-top:5px; display:none;">
                        <div id="progress-bar" style="width:0%; height:100%; background:#10b981; border-radius:2px; transition:width 0.2s;"></div>
                    </div>
                    <div id="step-info" style="font-size:12px; font-weight:normal; margin-top:2px;"></div>
                </div>
            </div>

            <div class="label" style="margin-top:15px;">Attention Rules</div>
            <div class="attn-container">
                <div class="attn-header" onclick="window.AttnManager.toggleView()">
                    <span style="display:flex; align-items:center;">
                        <span id="attn-caret" class="attn-caret closed">▼</span> 
                        Matrix (Query → Key)
                    </span>
                    <span style="cursor:pointer; text-decoration:underline" onclick="event.stopPropagation(); window.AttnManager.refresh(true)">Reset Default</span>
                </div>
                <div id="attn-matrix-root" class="attn-table-wrap collapsed">
                    </div>
            </div>
            
            <div class="label" style="margin-top:10px;">Settings</div>
            <div style="margin-bottom:10px; background:#f9fafb; padding:10px; border-radius:6px; border:1px solid #e5e7eb;">
                <div style="display:flex; justify-content:space-between; align-items:center; font-size:12px; color:#374151; margin-bottom:8px;">
                    <span style="font-weight:600;">Total Steps</span>
                    <input type="number" id="steps-input" min="1" max="100" value="28" style="width:60px; padding:4px; border:1px solid #d1d5db; border-radius:4px; text-align:right;">
                </div>
                
                <div style="display:flex; justify-content:space-between; align-items:center; font-size:12px; color:#374151; margin-bottom:8px;">
                    <div style="display:flex; flex-direction:column;">
                        <span style="font-weight:600;">Rule Switch Ratio</span>
                        <span style="font-size:9px; color:#9ca3af;">Custom Rules steps ratio (0-1), custom rules will be used before this step</span>
                    </div>
                    <input type="number" id="switch-step-input" min="0" max="1.0" step="0.1" value="0.0" style="width:60px; padding:4px; border:1px solid #d1d5db; border-radius:4px; text-align:right;">
                </div>

                <div style="display:flex; justify-content:space-between; align-items:center; font-size:12px; color:#374151;">
                    <div style="display:flex; flex-direction:column;">
                        <span style="font-weight:600;">Expand Value</span>
                        <span style="font-size:9px; color:#9ca3af;">Bbox expansion ratio during actual editing</span>
                    </div>
                    <input type="number" id="expand-input" min="0" max="1.0" step="0.05" value="0.15" style="width:60px; padding:4px; border:1px solid #d1d5db; border-radius:4px; text-align:right;">
                </div>
            </div>

            <button id="btn-edit" class="btn btn-success" onclick="window.App.runEdit()" disabled>🚀 Run Editing</button>
        </div>
    </div>

<script>

const HAS_IMAGE_PROMPT = true;
// ================= Attention Rule Manager =================
window.AttnManager = {
    components: [],
    rules: {},

    getLogicalGroup: function(name) {
        if (name.includes('Background')) return 'Background';
        if (name.startsWith('Main') || name.startsWith('Image P') || name.startsWith('Hint')) return name;
        const m = name.match(/(Noise|Image) BBox (\d+)/);
        return m ? `BBox ${m[2]}` : name;
    },

    isImageComp: function(group) {
        return group === 'Background' || group.startsWith('BBox');
    },

    checkDefaultRule: function(q, k, hasImagePrompt) {
        // Special override: Noise Background -> Noise BBox is False
        if (q === 'Noise Background' && k.startsWith('Noise BBox')) {
            return false; 
        }

        const qGroup = this.getLogicalGroup(q);
        const kGroup = this.getLogicalGroup(k);

        if (q === k) return true;

        if (hasImagePrompt) {
            if (qGroup === 'Image Prompt' || kGroup === 'Image Prompt') return true;
        }

        const qIsImg = this.isImageComp(qGroup);
        const kIsImg = this.isImageComp(kGroup);

        if (qIsImg && kIsImg) return true;

        if ((qIsImg && kGroup === 'Main Prompt') || (qGroup === 'Main Prompt' && kIsImg)) return true;

        if (qGroup.startsWith('Hint')) {
            if (kIsImg) return true;
        }
        if (kGroup.startsWith('Hint')) {
             if (qGroup === `BBox ${kGroup.split(' ')[1]}`) return true;
        }

        return false;
    },

    toggleView: function() {
        const root = document.getElementById('attn-matrix-root');
        const caret = document.getElementById('attn-caret');
        root.classList.toggle('collapsed');
        caret.classList.toggle('closed');
    },

    refresh: function(forceReset = false) {
        const boxes = window.EditorApp.state.boxes;
        const num = boxes.length;
        const hasImagePrompt = (typeof HAS_IMAGE_PROMPT !== 'undefined') ? HAS_IMAGE_PROMPT : true;

        let comps = ['Main Prompt'];
        if (hasImagePrompt) comps.push('Image Prompt');
        
        for(let i=1; i<=num; i++) {
            comps.push(`Hint ${i}`);
            comps.push(`Noise BBox ${i}`);
            comps.push(`Image BBox ${i}`);
        }
        comps.push('Noise Background');
        comps.push('Image Background');
        
        this.components = comps;

        if(forceReset) {
            this.rules = {};
            comps.forEach(q => comps.forEach(k => {
                this.rules[`${q}|${k}`] = this.checkDefaultRule(q, k, hasImagePrompt);
            }));
        } else {
            const oldRules = this.rules;
            this.rules = {};
            
            comps.forEach(q => comps.forEach(k => {
                const key = `${q}|${k}`;
                if (oldRules[key] !== undefined) {
                    this.rules[key] = oldRules[key];
                } else {
                    this.rules[key] = this.checkDefaultRule(q, k, hasImagePrompt);
                }
            }));
        }
        this.render();
    },

    toggle: function(q, k) {
        const key = `${q}|${k}`;
        this.rules[key] = !this.rules[key];
        this.render(); 
    },

    getShortLabel: function(name) {
        if(name === 'Main Prompt') return 'Main';
        if(name === 'Image Prompt') return 'ImgP';
        if(name === 'Noise Background') return 'N.BG';
        if(name === 'Image Background') return 'I.BG';
        return name.replace('Hint ', 'H')
                   .replace('Noise BBox ', 'N.B')
                   .replace('Image BBox ', 'I.B');
    },

    render: function() {
        const root = document.getElementById('attn-matrix-root');
        if(!root) return;
        
        let html = '<table class="attn-table"><thead><tr><th style="z-index:20; left:0;">Q \\ K</th>';
        this.components.forEach(c => {
            html += `<th>${this.getShortLabel(c)}</th>`;
        });
        html += '</tr></thead><tbody>';

        this.components.forEach(q => {
            html += `<tr><td class="row-label" title="${q}">${this.getShortLabel(q)}</td>`;
            this.components.forEach(k => {
                const val = this.rules[`${q}|${k}`];
                const icon = val ? '<span class="attn-v">●</span>' : '<span class="attn-x">×</span>';
                const bgStyle = val ? 'background:rgba(16, 185, 129, 0.1);' : ''; 
                
                html += `<td style="${bgStyle}"><div class="attn-cell" onclick="window.AttnManager.toggle('${q}','${k}')" title="${q} -> ${k}">` + 
                        icon + 
                        `</div></td>`;
            });
            html += '</tr>';
        });
        html += '</tbody></table>';
        root.innerHTML = html;
    },

    getRulesDict: function() { return this.rules; }
};

// ================= Configuration =================
const EXAMPLES = [
    { src: "./custom_assets/cup.png", text: "Replace the glass that has been used and left on the desk with a small potted plant" },
    { src: "./custom_assets/crowd.png", text: "Find the woman with light blue backpack and change the color of her shoes to red" },
    { src: "./custom_assets/festival.png", text: "The sun has now set, and someone has decorated the large bush on the right with colorful fairy lights for the evening's festivities." },
    { src: "./custom_assets/cream.png", text: "Replace the spread on the left cracker with peanut butter and the spread on the right cracker with avocado cream." },
    { src: "./custom_assets/keyboard.png", text: "Change the color of the keyboard with yellow sticky notes to black." },
    { src: "./custom_assets/sunglasses.png", text: "Remove the sunglasses from the person sitting on the left." },
    { src: "./custom_assets/snowman.png", text: "Replace the individual whose attire suggests a greater emphasis on thermal insulation for cold weather with a snowman." },
    { src: "./custom_assets/poster.png", text: "Change the single word at the top of the elephant coloring page, which is written in all caps with a thick black outline and white fill, to 'GIRAFFE'." },
    { src: "./custom_assets/slides.png", text: "Change the closing date for the survey, to reflect an extension of one month due to unexpectedly low initial participation rates." },
];

// ================= Editor Logic =================
window.EditorApp = {
    COLORS: ['#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4', '#3b82f6', '#a855f7', '#ec4899'],
    state: { 
        boxes: [], selectedId: null, mode: 'idle', 
        startX: 0, startY: 0, activeBoxEl: null, 
        imgNaturalWidth: 0, imgNaturalHeight: 0, boxIdCounter: 1,
        resizeHandle: null, initialBoxData: null
    },
    dom: {},

    init: function() {
        const targetImg = document.getElementById('target-img');
        if (!targetImg) return false;

        this.dom = {
            targetImg: targetImg, editorStage: document.getElementById('editor-stage'), overlay: document.getElementById('overlay'),
            emptyState: document.getElementById('empty-state'), regionList: document.getElementById('region-list'),
            globalInput: document.getElementById('global-prompt'),
            selectionPanel: document.getElementById('selection-panel'),
            labelInput: document.getElementById('label-input'),
            colorRow: document.getElementById('color-row')
        };
        
        // Init Colors
        if(this.dom.colorRow) {
            this.dom.colorRow.innerHTML = this.COLORS.map(c => 
                `<div class="color-opt" style="background-color:${c}" onclick="window.EditorApp.updateColor('${c}')"></div>`
            ).join('');
        }

        if(this.dom.labelInput) {
            this.dom.labelInput.oninput = (e) => {
                if(this.state.selectedId) {
                    const box = this.state.boxes.find(b => b.id === this.state.selectedId);
                    if(box) {
                        box.label = e.target.value;
                        this.updateBoxVisual(box);
                        this.renderList();
                    }
                }
            };
        }
        
        // Sync Instruction to Global Prompt if empty
        const mainInstr = document.getElementById('instruction');
        if (mainInstr && this.dom.globalInput) {
             mainInstr.addEventListener('input', (e) => {
                 if(this.state.boxes.length === 0) {
                     this.dom.globalInput.value = e.target.value;
                 }
             });
        }

        this.setupEvents();
        return true;
    },

    loadData: function(base64Img, xmlData) {
        if (!this.dom.targetImg && !this.init()) {
            setTimeout(() => this.loadData(base64Img, xmlData), 200); return;
        }
        if (base64Img) {
            this.dom.targetImg.src = base64Img;
            this.dom.targetImg.onload = () => {
                this.state.imgNaturalWidth = this.dom.targetImg.naturalWidth;
                this.state.imgNaturalHeight = this.dom.targetImg.naturalHeight;
                this.dom.emptyState.classList.add('hidden');
                this.dom.editorStage.classList.remove('hidden');
                this.refreshOverlay();
                window.AttnManager.refresh(true); // <--- Matrix Refresh
            };
        }
        if (xmlData) {
            this.parseXML(xmlData);
        } else {
            // New image loaded: reset boxes
            this.state.boxes = [];
            this.dom.overlay.innerHTML = '';
            // Default Global Prompt to Instruction Value
            const instr = document.getElementById('instruction').value;
            if(this.dom.globalInput) this.dom.globalInput.value = instr || "keep remaining parts unchanged";
            this.renderList();
        }
        this.deselectAll();
    },

    getData: function() {
        const regionData = this.state.boxes.map(b => ({
            bbox_2d: [Math.round(b.realRect[0]), Math.round(b.realRect[1]), Math.round(b.realRect[2]), Math.round(b.realRect[3])],
            hint: b.label
        }));
        
        let globalPrompt = this.dom.globalInput ? this.dom.globalInput.value : "";
        
        return `<gen_image>${globalPrompt}</gen_image><region>${JSON.stringify(regionData)}</region>`;
    },

    parseXML: function(text) {
        const globalMatch = text.match(/<gen_image>(.*?)<\/gen_image>/s) || text.match(/<global>(.*?)<\/global>/s);
        if (globalMatch && this.dom.globalInput) {
            this.dom.globalInput.value = globalMatch[1];
        }
        
        // Parse Region
        const regionMatch = text.match(/<region>(.*?)<\/region>/s);
        if (regionMatch) {
            try {
                const regions = JSON.parse(regionMatch[1]);
                this.state.boxes = [];
                this.dom.overlay.innerHTML = '';
                regions.forEach((r, idx) => {
                    const [x1, y1, x2, y2] = r.bbox_2d;
                    const newBox = {
                        id: this.state.boxIdCounter++,
                        realRect: [x1, y1, x2, y2],
                        label: r.hint || "Region",
                        color: this.COLORS[idx % this.COLORS.length]
                    };
                    this.state.boxes.push(newBox);
                    this.createBoxElement(newBox);
                });
                this.renderList();
                this.refreshOverlay();
                window.AttnManager.refresh(true); // <--- Matrix Refresh
            } catch (e) { console.error("Parse Error", e); }
        }
    },

    createBoxElement: function(box) {
        const el = document.createElement('div');
        el.className = 'bbox';
        el.dataset.id = box.id;
        el.innerHTML = `
            <div class="bbox-label" style="background-color:${box.color}">${box.label}</div>
            <div class="resize-handle handle-nw" data-h="nw"></div>
            <div class="resize-handle handle-ne" data-h="ne"></div>
            <div class="resize-handle handle-sw" data-h="sw"></div>
            <div class="resize-handle handle-se" data-h="se"></div>
        `;
        this.dom.overlay.appendChild(el);
        this.updateBoxVisual(box);
    },

    updateBoxVisual: function(box) {
        const el = this.dom.overlay.querySelector(`.bbox[data-id="${box.id}"]`);
        if (!el || !this.state.imgNaturalWidth) return;
        const rect = this.dom.targetImg.getBoundingClientRect();
        const scaleX = rect.width / this.state.imgNaturalWidth;
        const scaleY = rect.height / this.state.imgNaturalHeight;
        
        const [x1, y1, x2, y2] = box.realRect;
        const sx = x1 * scaleX;
        const sy = y1 * scaleY;
        const sw = (x2 - x1) * scaleX;
        const sh = (y2 - y1) * scaleY;
        
        el.style.left = sx + 'px';
        el.style.top = sy + 'px';
        el.style.width = sw + 'px';
        el.style.height = sh + 'px';
        el.style.borderColor = box.color;
        
        const labelEl = el.querySelector('.bbox-label');
        if(labelEl) {
            labelEl.innerText = box.label;
            labelEl.style.backgroundColor = box.color;
            if (sy < 30) {
                labelEl.style.top = "0px";
                labelEl.style.borderRadius = "0 0 4px 0";
            } else {
                labelEl.style.top = "-24px";
                labelEl.style.borderRadius = "4px 4px 0 0";
            }
        }
    },

    refreshOverlay: function() {
        this.state.boxes.forEach(b => this.updateBoxVisual(b));
    },

    renderList: function() {
        this.dom.regionList.innerHTML = '';
        this.state.boxes.slice().reverse().forEach(box => {
            const item = document.createElement('div');
            item.className = 'list-item';
            if (this.state.selectedId === box.id) item.classList.add('active');
            item.innerHTML = `<div class="color-dot" style="background-color:${box.color}"></div><div class="item-label" title="${box.label}">${box.label}</div>`;
            item.onclick = () => this.selectBox(box.id);
            this.dom.regionList.appendChild(item);
        });
        document.getElementById('region-count').innerText = this.state.boxes.length;
    },
    
    selectBox: function(id) {
        this.state.selectedId = id;
        const all = this.dom.overlay.querySelectorAll('.bbox');
        all.forEach(el => el.classList.remove('selected'));
        const el = this.dom.overlay.querySelector(`.bbox[data-id="${id}"]`);
        if(el) el.classList.add('selected');
        
        if(this.dom.selectionPanel) this.dom.selectionPanel.classList.remove('disabled');
        const box = this.state.boxes.find(b => b.id === id);
        if(box && this.dom.labelInput) this.dom.labelInput.value = box.label;
        
        this.renderList();
    },
    
    deselectAll: function() {
        this.state.selectedId = null;
        const all = this.dom.overlay.querySelectorAll('.bbox');
        all.forEach(el => el.classList.remove('selected'));
        if(this.dom.selectionPanel) this.dom.selectionPanel.classList.add('disabled');
        if(this.dom.labelInput) this.dom.labelInput.value = "";
        this.renderList();
    },

    deleteSelected: function() {
        if(this.state.selectedId) {
            const el = this.dom.overlay.querySelector(`.bbox[data-id="${this.state.selectedId}"]`);
            if(el) el.remove();
            this.state.boxes = this.state.boxes.filter(b => b.id !== this.state.selectedId);
            this.deselectAll();
            this.renderList();
            window.AttnManager.refresh(); // <--- Matrix Refresh
        }
    },
    
    updateColor: function(color) {
        if(this.state.selectedId) {
            const box = this.state.boxes.find(b => b.id === this.state.selectedId);
            if(box) {
                box.color = color;
                this.updateBoxVisual(box);
                this.renderList();
            }
        }
    },

    setupEvents: function() {
        this.dom.overlay.onmousedown = (e) => {
            const rect = this.dom.overlay.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            if(e.target.classList.contains('resize-handle')) {
                const boxEl = e.target.closest('.bbox');
                const id = parseInt(boxEl.dataset.id);
                this.state.selectedId = id;
                this.state.mode = 'resizing';
                this.state.resizeHandle = e.target.dataset.h;
                this.state.initialBoxData = JSON.parse(JSON.stringify(this.state.boxes.find(b => b.id === id)));
                this.state.startX = x; 
                this.state.startY = y;
                e.stopPropagation();
                return;
            }

            const clickedBox = e.target.closest('.bbox');
            if (clickedBox) {
                const id = parseInt(clickedBox.dataset.id);
                this.selectBox(id);
                this.state.mode = 'dragging';
                this.state.startX = x;
                this.state.startY = y;
                this.state.initialBoxData = JSON.parse(JSON.stringify(this.state.boxes.find(b => b.id === id)));
                e.stopPropagation();
                return;
            }

            if(e.target === this.dom.overlay) {
                this.deselectAll();
                this.state.mode = 'drawing';
                this.state.startX = x;
                this.state.startY = y;
                this.state.activeBoxEl = document.createElement('div');
                Object.assign(this.state.activeBoxEl.style, {
                    position: 'absolute', border: '2px dashed #4f46e5',
                    left: x+'px', top: y+'px', width: '0', height: '0', pointerEvents: 'none'
                });
                this.dom.overlay.appendChild(this.state.activeBoxEl);
            }
        };

        window.addEventListener('mousemove', (e) => {
            if (this.state.mode === 'idle') return;
            const rect = this.dom.overlay.getBoundingClientRect();
            let x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
            let y = Math.max(0, Math.min(e.clientY - rect.top, rect.height));
            const scaleX = this.state.imgNaturalWidth / rect.width;
            const scaleY = this.state.imgNaturalHeight / rect.height;

            if (this.state.mode === 'drawing' && this.state.activeBoxEl) {
                const w = x - this.state.startX;
                const h = y - this.state.startY;
                this.state.activeBoxEl.style.left = (w < 0 ? x : this.state.startX) + 'px';
                this.state.activeBoxEl.style.top = (h < 0 ? y : this.state.startY) + 'px';
                this.state.activeBoxEl.style.width = Math.abs(w) + 'px';
                this.state.activeBoxEl.style.height = Math.abs(h) + 'px';
            } 
            else if (this.state.mode === 'dragging' && this.state.selectedId) {
                const box = this.state.boxes.find(b => b.id === this.state.selectedId);
                const dx = (x - this.state.startX) * scaleX;
                const dy = (y - this.state.startY) * scaleY;
                const [ix1, iy1, ix2, iy2] = this.state.initialBoxData.realRect;
                let nx1 = Math.max(0, Math.min(ix1 + dx, this.state.imgNaturalWidth - (ix2 - ix1)));
                let ny1 = Math.max(0, Math.min(iy1 + dy, this.state.imgNaturalHeight - (iy2 - iy1)));
                box.realRect = [nx1, ny1, nx1 + (ix2 - ix1), ny1 + (iy2 - iy1)];
                this.updateBoxVisual(box);
            }
            else if (this.state.mode === 'resizing' && this.state.selectedId) {
                const box = this.state.boxes.find(b => b.id === this.state.selectedId);
                const realX = x * scaleX;
                const realY = y * scaleY;
                let [x1, y1, x2, y2] = this.state.initialBoxData.realRect;
                const h = this.state.resizeHandle;
                if (h.includes('w')) x1 = Math.min(realX, x2 - 10);
                if (h.includes('e')) x2 = Math.max(realX, x1 + 10);
                if (h.includes('n')) y1 = Math.min(realY, y2 - 10);
                if (h.includes('s')) y2 = Math.max(realY, y1 + 10);
                x1 = Math.max(0, x1); y1 = Math.max(0, y1);
                x2 = Math.min(this.state.imgNaturalWidth, x2); y2 = Math.min(this.state.imgNaturalHeight, y2);
                box.realRect = [x1, y1, x2, y2];
                this.updateBoxVisual(box);
            }
        });

        window.addEventListener('mouseup', (e) => {
            if (this.state.mode === 'drawing') {
                if(this.state.activeBoxEl) {
                    const domRect = this.state.activeBoxEl.getBoundingClientRect();
                    const overlayRect = this.dom.overlay.getBoundingClientRect();
                    this.state.activeBoxEl.remove();
                    this.state.activeBoxEl = null;
                    if (domRect.width > 5 && domRect.height > 5) {
                        const scaleX = this.state.imgNaturalWidth / overlayRect.width;
                        const scaleY = this.state.imgNaturalHeight / overlayRect.height;
                        const x1 = (domRect.left - overlayRect.left) * scaleX;
                        const y1 = (domRect.top - overlayRect.top) * scaleY;
                        const x2 = x1 + (domRect.width * scaleX);
                        const y2 = y1 + (domRect.height * scaleY);
                        const newBox = {
                            id: this.state.boxIdCounter++,
                            realRect: [x1, y1, x2, y2],
                            label: "New Region",
                            color: this.COLORS[this.state.boxes.length % this.COLORS.length]
                        };
                        this.state.boxes.push(newBox);
                        this.createBoxElement(newBox);
                        this.selectBox(newBox.id);
                        window.AttnManager.refresh(); // <--- Matrix Refresh
                    }
                }
            }
            this.state.mode = 'idle';
        });
        window.addEventListener('resize', () => this.refreshOverlay());
    }
};

// ================= Main App Logic =================
window.App = {
    currentFile: null,
    init: function() { this.renderExamples(); },
    renderExamples: function() {
        const list = document.getElementById('example-list');
        if (!list) return;
        list.innerHTML = "";
        EXAMPLES.forEach(ex => {
            const div = document.createElement('div');
            div.className = "ex-item";
            div.innerHTML = `<img src="${ex.src}" alt="example"><div class="ex-desc">${ex.text}</div>`;
            div.onclick = async () => {
                try {
                    const res = await fetch(ex.src);
                    const blob = await res.blob();
                    const file = new File([blob], "example.png", { type: blob.type });
                    this.loadFile(file);
                    document.getElementById('instruction').value = ex.text;
                } catch(e) { alert("Failed to load example: " + e.message); }
            };
            list.appendChild(div);
        });
    },
    loadFile: function(file) {
        this.currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('input-preview').src = e.target.result;
            document.getElementById('input-preview').style.display = 'block';
            document.getElementById('upload-text').style.display = 'none';
            if (window.EditorApp) window.EditorApp.loadData(e.target.result, null);
            document.getElementById('btn-edit').disabled = false;
        };
        reader.readAsDataURL(file);
    },
    runPlan: async function() {
        if (!this.currentFile) return alert("Please upload an image first.");
        const instr = document.getElementById('instruction').value;
        this.setLoading(true);
        const fd = new FormData();
        fd.append('image', this.currentFile); fd.append('instruction', instr);
        try {
            const res = await fetch('/api/plan', { method: 'POST', body: fd });
            const data = await res.json();
            if (window.EditorApp) window.EditorApp.loadData(data.image_base64, data.xml);
        } catch(e) { alert(e.message); } finally { this.setLoading(false); }
    },
runEdit: async function() {
        const globalPromptEl = document.getElementById('global-prompt');
        if (!globalPromptEl || !globalPromptEl.value.trim()) return alert("⚠️ Global Prompt cannot be empty.");
        
        this.setLoading(true); 
        this.resetProgress();
        
        const runId = "run_" + Date.now(); 
        this.startPolling(runId);
        
        const fd = new FormData();
        fd.append('image', this.currentFile);
        fd.append('instruction', document.getElementById('instruction').value || "");
        fd.append('xml', window.EditorApp.getData());
        fd.append('num_inference_steps', document.getElementById('steps-input').value);
        fd.append('run_id', runId);
        fd.append('attention_rules', JSON.stringify(window.AttnManager.getRulesDict())); 
        
        const switchStep = document.getElementById('switch-step-input').value;
        fd.append('attention_switch_step', switchStep);

        const expandVal = document.getElementById('expand-input').value;
        fd.append('expand_value', expandVal);

        try {
            const res = await fetch('/api/edit', { method: 'POST', body: fd });
            if (!res.ok) {
                const errorText = await res.text(); 
                throw new Error(`Server Error (${res.status}): ${errorText}`);
            }
            const data = await res.json();
            const img = document.getElementById('result-img');
            img.src = data.result_base64; 
            img.style.display = 'block';
            document.getElementById('result-placeholder').style.display = 'none';
        } catch(e) { 
            alert(e.message); 
        } finally { 
            this.setLoading(false); 
            this.stopPolling(); 
        }
    },
    pollInterval: null,
    startPolling: function(runId) {
        if(this.pollInterval) clearInterval(this.pollInterval);
        this.pollInterval = setInterval(async () => {
            try {
                const res = await fetch('/api/progress?id=' + runId);
                const data = await res.json();
                if(data.progress) this.updateProgress(data);
            } catch(e) {}
        }, 500);
    },
    stopPolling: function() { clearInterval(this.pollInterval); this.pollInterval = null; },
    resetProgress: function() {
        document.getElementById('progress-bar').style.width = '0%';
        document.getElementById('progress-bar-container').style.display = 'block';
    },
    updateProgress: function(data) {
        document.getElementById('progress-bar').style.width = Math.round(data.progress * 100) + '%';
        document.getElementById('step-info').innerText = `Step ${data.step} / ${data.total}`;
    },
    setLoading: function(isLoading) {
        document.getElementById('loading-mask').style.display = isLoading ? 'flex' : 'none';
        document.getElementById('btn-plan').disabled = isLoading;
        document.getElementById('btn-edit').disabled = isLoading;
    }
};
window.App.init();
</script>
</body>
</html>
"""

for k, v in ICONS.items():
    FULL_PAGE_TEMPLATE = FULL_PAGE_TEMPLATE.replace(f"[[ICON_{k.upper()}]]", v)


pipeline = None
output_base_dir = 'output/gradio_custom'
DEFAULT_STEPS = 28

def main():
    global pipeline, output_base_dir, DEFAULT_STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_type", default="flux", choices=["flux", "qwen"])
    parser.add_argument("--vlm_ckpt_path", default="TainU/RePlan-Qwen2.5-VL-7B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default='./output/gradio_custom')
    parser.add_argument("--server_port", type=int, default=8080)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--diffusion_model_name", default=None)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--vlm_prompt_template_path", default="replan.txt")
    parser.add_argument("--dtype", default="bf16")
    args = parser.parse_args()

    if args.pipeline_type == "flux": DEFAULT_STEPS = 28
    else: DEFAULT_STEPS = 50

    if not os.path.isabs(args.output_dir): output_base_dir = os.path.join(project_root, args.output_dir)
    else: output_base_dir = args.output_dir
    os.makedirs(output_base_dir, exist_ok=True)

    if pipeline is None:
        print(f"Initializing RePlanPipeline ({args.pipeline_type})...")
        pipeline = RePlanPipeline(
            vlm_ckpt_path=args.vlm_ckpt_path,
            diffusion_model_name=args.diffusion_model_name,
            pipeline_type=args.pipeline_type,
            output_dir=output_base_dir,
            device=args.device,
            torch_dtype=_parse_torch_dtype(args.dtype),
            vlm_prompt_template_path=args.vlm_prompt_template_path,
            lora_path=args.lora_path,
            init_vlm=True,
            image_dir=project_root,
        )

    block_css = """
    .gradio-container { padding: 0 !important; margin: 0 !important; width: 100% !important; max-width: 100% !important; height: 100vh !important; overflow: hidden; }
    footer { display: none !important; }
    iframe { width: 100%; height: 100vh; border: none; display: block; }
    """
        
    with gr.Blocks(title="RePlan Pro", css=block_css) as demo:
        gr.HTML('<iframe src="/custom_ui"></iframe>')

    def mount_custom_routes(app: FastAPI):
        assets_path = os.path.join(project_root, "assets")
        if os.path.exists(assets_path): app.mount("/custom_assets", StaticFiles(directory=assets_path), name="assets")
        

        @app.get("/custom_ui", response_class=HTMLResponse)
        async def serve_ui():
            html_content = FULL_PAGE_TEMPLATE.replace('value="28"', f'value="{DEFAULT_STEPS}"')
            
            current_type = getattr(pipeline, "pipeline_type", "flux")
            js_bool = "true" if current_type == "qwen" else "false"
            
            html_content = html_content.replace(
                "const HAS_IMAGE_PROMPT = true;", 
                f"const HAS_IMAGE_PROMPT = {js_bool};"
            )
            
            default_ratio = "0.5" if current_type == "qwen" else "0.0"
            html_content = html_content.replace('value="0.0"', f'value="{default_ratio}"')
            
            default_expand = "0.05" if current_type == "qwen" else "0.15"
            html_content = html_content.replace('value="0.15"', f'value="{default_expand}"')
            
            return html_content

        @app.get("/api/progress")
        async def api_progress(id: str): return JSONResponse(PROGRESS_STATES.get(id, {"progress": 0, "step": 0}))

        @app.post("/api/plan")
        async def api_plan(image: UploadFile = File(...), instruction: str = Form(...)):
            with gpu_lock:
                run_id = f"plan_{datetime.now().strftime('%H%M%S')}"
                temp_path = os.path.join(output_base_dir, f"{run_id}.png")
                with open(temp_path, "wb") as f: f.write(await image.read())
                res = pipeline.get_vlm_response(instruction, temp_path)
                return JSONResponse({"xml": res[0] if isinstance(res, tuple) else res, "image_base64": image_to_base64(temp_path)})

        @app.post("/api/edit")
        async def api_edit(
            image: UploadFile = File(...), 
            xml: str = Form(...),
            num_inference_steps: int = Form(DEFAULT_STEPS),
            run_id: str = Form(None),
            attention_rules: str = Form(None),
            attention_switch_step: float = Form(0.0),
            expand_value: float = Form(0.15)
        ):
            if not run_id: run_id = f"edit_{datetime.now().strftime('%H%M%S')}"
            
            def callback(pipe, step_index, timestep, callback_kwargs):
                PROGRESS_STATES[run_id] = {"progress": min((step_index+1)/num_inference_steps, 1.0), "step": step_index+1, "total": num_inference_steps}
                return callback_kwargs

            current_type = getattr(pipeline, "pipeline_type", "flux")
            use_image_prompt = (current_type == "qwen")

            custom_rules_map = {}
            if attention_rules:
                try:
                    raw = json.loads(attention_rules)
                    custom_rules_map = { tuple(k.split('|')): v for k, v in raw.items() }
                except Exception as e:
                    print(f"Error parsing custom rules: {e}")

            num_regions = 0
            try:
                region_match = re.search(r'<region>(.*?)</region>', xml, re.DOTALL)
                if region_match:
                    regions = json.loads(region_match.group(1))
                    num_regions = len(regions)
            except: pass

            default_rules_map = generate_default_split_rules(num_regions, has_image_prompt=use_image_prompt)

            step_attention_dict = {}
            
            if attention_switch_step <= 1.0:
                switch_at = int(attention_switch_step * num_inference_steps)
            else:
                switch_at = int(attention_switch_step)
                
            switch_at = max(0, min(switch_at, num_inference_steps))
            
            for step in range(num_inference_steps):
                if step < switch_at:
                    step_attention_dict[step] = custom_rules_map if custom_rules_map else default_rules_map
                else:
                    step_attention_dict[step] = default_rules_map

            with gpu_lock:
                try:
                    temp_path = os.path.join(output_base_dir, f"{run_id}_src.png")
                    with open(temp_path, "wb") as f: f.write(await image.read())
                    
                    print(f"[API Edit] Steps: {num_inference_steps} | Switch Logic: {attention_switch_step} -> Step {switch_at}")

                    res = pipeline.region_edit_with_attention(
                        image=temp_path,
                        instruction="",
                        response=xml,
                        output_dir=os.path.join(output_base_dir, run_id),
                        attention_rules=step_attention_dict, 
                        expand_value=expand_value,
                        pipeline_kwargs={
                            "num_inference_steps": num_inference_steps,
                            "callback_on_step_end": callback
                        }
                    )
                    return JSONResponse({"result_base64": image_to_base64(res[0] if isinstance(res, tuple) else res)})
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return JSONResponse({"error": str(e)}, status_code=500)
                finally:
                    if run_id in PROGRESS_STATES: del PROGRESS_STATES[run_id]

    demo.launch(server_name="0.0.0.0", server_port=args.server_port, share=args.share, prevent_thread_lock=True)
    if demo.app: mount_custom_routes(demo.app)
    demo.block_thread()

if __name__ == "__main__":
    main()