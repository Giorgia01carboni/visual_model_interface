'''
Launch with: python -m virtual_assistant.app
'''

import gradio as gr
import os
import plotly.graph_objects as go
import traceback

# Import from the local package
from .service import SpatialService 
from .llm_service import LLMService
from .utils import create_plot # Import utils for direct plotting

# --- CONFIGURATION ---
MODEL_PATH = "models/spatiallm_checkpoint" 
OUTPUT_DIR = "output"
DEFAULT_PCD = "examples/industrial_scene.ply"

# --- INITIALIZE SERVICES ---
print("--- SYSTEM STARTUP ---")
try:
    llm = LLMService(api_key=os.environ.get("OPENAI_API_KEY"))
    print("LLM Service initialized.")
except Exception as e:
    print(f"LLM Service FAILED: {e}")

try:
    print(f"Loading SpatialLM from {MODEL_PATH}...")
    service = SpatialService(MODEL_PATH) 
    print("SpatialService initialized.")
except Exception as e:
    print(f"SpatialService FAILED: {e}")

# --- HELPER: RENDER SCENE ---
def update_view(file_obj, density, point_size, boxes):
    """Refreshes the plot using current settings and existing boxes."""
    if not file_obj:
        return None
    
    try:
        # Load raw points
        _, points, colors, _ = service.preprocess(file_obj.name)
        
        # Create plot with client-side toggle (from utils.py)
        fig = create_plot(
            points, colors, 
            boxes=boxes, 
            max_points=density, 
            point_size=point_size
        )
        return fig
    except Exception as e:
        print(f"Render Error: {e}")
        return None

# --- ORCHESTRATOR LOGIC ---
def orchestrator(message, history, file_obj, density, point_size, top_k, top_p, temp, beams, saved_plot, saved_code, saved_boxes):
    print(f"\n[DEBUG] New Message: '{message}'")

    current_plot = saved_plot
    current_code = saved_code
    current_boxes = saved_boxes if saved_boxes else []
    
    # 1. HISTORY SANITIZER
    clean_history = []
    if history:
        for entry in history:
            if isinstance(entry, (list, tuple)):
                clean_history.append({"role": "user", "content": str(entry[0])})
                if len(entry) > 1 and entry[1] is not None:
                    clean_history.append({"role": "assistant", "content": str(entry[1])})
            elif isinstance(entry, dict):
                clean_history.append(entry)
    history = clean_history

    # 2. File Check
    if not file_obj:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ Please upload a .ply file first."})
        # Output: Chat, Plot, Code, StatePlot, StateCode, StateBoxes
        yield history, saved_plot, saved_code, saved_plot, saved_code, saved_boxes
        return

    try:
        # 3. Ask LLM
        print("[DEBUG] 🧠 Calling OpenAI...")
        decision = llm.get_response(history, message)
        
        history.append({"role": "user", "content": message})

        # --- SCENARIO A: Text Response (NO RENDER) ---
        if decision["type"] == "text":
            history.append({"role": "assistant", "content": decision["content"]})
            # FIX: Yield existing 'saved_plot' instead of gr.no_update()
            yield history, saved_plot, saved_code, saved_plot, saved_code, saved_boxes
        
        # --- SCENARIO B: Tool Call ---
        elif decision["type"] == "tool_list":
            
            bot_response = "Processing request...\n\n"
            history.append({"role": "assistant", "content": bot_response})
            # Intermediate yield: Keep everything as is
            yield history, saved_plot, saved_code, saved_plot, saved_code, saved_boxes
            
            new_found_boxes = [] 
            temp_layout_str = ""
            total_found_count = 0

            # --- ACTION LOOP ---
            for i, action in enumerate(decision["actions"]):
                cats = action["categories"]
                d_type = action["detect_type"]
                
                step_msg = f"**Step {i+1}**: Scanning for **{d_type}** ({', '.join(cats)})...\n"
                bot_response += step_msg
                history[-1]["content"] = bot_response
                # Update chat, keep visuals static
                yield history, saved_plot, saved_code, saved_plot, saved_code, saved_boxes
                
                # 4. CALL INFERENCE
                result = service.predict(
                    file_path=file_obj.name, 
                    categories=cats, 
                    detect_type=d_type,
                    density=density if i == 0 else 0,
                    point_size=point_size, 
                    top_k=top_k, 
                    top_p=top_p, 
                    temperature=temp, 
                    num_beams=beams
                )
                
                if result["success"]:
                    count = result["count"]
                    layout_code = result["layout_str"]
                    boxes = result.get("boxes", [])
                    
                    total_found_count += count
                    
                    if count > 0:
                        temp_layout_str += f"# --- Found {count} {', '.join(cats)} ---\n{layout_code}\n\n"
                        new_found_boxes.extend(boxes)
                    
                    bot_response += f"   Found **{count}** instances.\n"
                else:
                    bot_response += f"  Error: {result['error']}\n"
                
                history[-1]["content"] = bot_response
                # Update chat, keep visuals static
                yield history, saved_plot, saved_code, saved_plot, saved_code, saved_boxes

            # Did we find anything new?
            if total_found_count > 0:
                bot_response += f"\n✅ **Result:** Found {total_found_count} items."
                
                # Update State with NEW boxes
                current_boxes = new_found_boxes 
                current_code = temp_layout_str
                
                # Generate Unified Plot (Only re-render here)
                current_plot = update_view(file_obj, density, point_size, current_boxes)
                
            else:
                bot_response += f"\n❌ **Result:** No instances found. Keeping previous view."
                # Keep old state
            
            history[-1]["content"] = bot_response
            
            # Return Final Updated State
            yield history, current_plot, current_code, current_plot, current_code, current_boxes

    except Exception as e:
        error_trace = traceback.format_exc()
        history.append({"role": "assistant", "content": f"❌ System Error:\n{str(e)}"})
        # On error: Show trace in code block, keep plot as is
        yield history, saved_plot, error_trace, saved_plot, saved_code, saved_boxes

# --- UI DEFINITION ---
with gr.Blocks(title="SpatialLM Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏭 SpatialLM Industrial Assistant")
    
    # --- GLOBAL STATE ---
    saved_plot = gr.State(value=None)
    saved_code = gr.State(value="")
    saved_boxes = gr.State(value=[]) 

    with gr.Row():
        with gr.Column(scale=1):
            input_file = gr.File(label="Upload Scene (.ply)", file_types=[".ply"])
            
            gr.Examples(
                examples=[[DEFAULT_PCD]], 
                inputs=input_file,
                label="Or try this example:"
            )

            chatbot = gr.Chatbot(height=400, label="Conversation")
            
            # --- VISUAL SETTINGS ---
            with gr.Accordion("Visual Settings", open=True):
                gr.Markdown("*Use the 'Show/Hide Boxes' buttons inside the 3D view (top left).*")
                with gr.Row():
                    point_density = gr.Slider(10000, 500000, 100000, label="Max Points")
                    point_size = gr.Slider(1.0, 10.0, 3.0, step=0.5, label="Point Size")

            msg = gr.Textbox(label="Your Request", placeholder="e.g., 'Find all pipes and boilers'")
            clear = gr.Button("Clear Chat")
            
            with gr.Accordion("Model Inference Parameters", open=False):
                top_k = gr.Slider(1, 100, 10, label="Top K")
                top_p = gr.Slider(0.1, 1.0, 0.95, label="Top P")
                temperature = gr.Slider(0.1, 2.0, 0.6, step=1, label="Temperature")
                num_beams = gr.Slider(1, 10, 2, step=1, label="Beams")

        with gr.Column(scale=2):
            output_plot = gr.Plot(label="3D Visualization")
            output_content = gr.Code(label="Layout Code", language="python")

    # --- EVENTS ---

    # 1. Chat Submission
    msg.submit(
        fn=orchestrator,
        inputs=[
            msg, chatbot, input_file, 
            point_density, point_size, 
            top_k, top_p, temperature, num_beams,
            saved_plot, saved_code, saved_boxes
        ],
        outputs=[
            chatbot, output_plot, output_content, 
            saved_plot, saved_code, saved_boxes
        ]
    )

    # 2. Immediate Render on File Upload
    def on_file_upload(file, density, size):
        fig = update_view(file, density, size, boxes=[])
        return fig, "", []

    input_file.change(
        fn=on_file_upload,
        inputs=[input_file, point_density, point_size],
        outputs=[output_plot, saved_code, saved_boxes]
    )

    # 3. Reactive Visual Settings (Sliders)
    def on_visual_change(file, density, size, boxes):
        return update_view(file, density, size, boxes)

    common_inputs = [input_file, point_density, point_size, saved_boxes]
    
    point_density.release(fn=on_visual_change, inputs=common_inputs, outputs=[output_plot])
    point_size.release(fn=on_visual_change, inputs=common_inputs, outputs=[output_plot])
    
    # 4. Clear
    clear.click(
        lambda: ([], None, "", None, "", []), 
        None, 
        [chatbot, output_plot, output_content, saved_plot, saved_code, saved_boxes], 
        queue=False
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True, allowed_paths=[OUTPUT_DIR, "/home/mradmin"])