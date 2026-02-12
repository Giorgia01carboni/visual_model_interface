import os 
from openai import OpenAI
import json

SPATIALLM_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "detect_3d_objects",
            "description": "Scans the 3D scene to detect, count, and locate objects. Can handle multiple categories in a single call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ALL object classes to find. IMPORTANT: Use full noun phrases (e.g. 'electric equipment', not just 'electric')."
                    },
                    "detect_type": {
                        "type": "string",
                        "enum": ["object", "arch", "all"],
                        "description": "Use 'object' for equipment, 'arch' for building structure, 'all' for everything."
                    }
                },
                "required": ["categories"]
            }
        }
    }
]

class LLMService:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def get_response(self, history, user_message):
        # --- SYSTEM PROMPT ---
        system_instruction = """
        You are a specialized industrial 3D analyst.
        
        PROTOCOL:
        1. **CHECK HISTORY FIRST**: If the user asks for a count or location of items that were **already found** in the recent conversation (e.g., "How many did you find?", "Where are they?"), DO NOT use the tool. Answer directly from the conversation history.
        
        2. **NEW SEARCHES**: Use the 'detect_3d_objects' tool ONLY if the user asks to find/count items that haven't been processed yet, or asks to "scan again".
        
        3. **GROUPING**: If the user asks for multiple objects, combine them into a SINGLE tool call (e.g., categories=['pipe', 'boiler']).
        
        4. **NAMING**: precision is key. If the user says "electric equipment", the category MUST be "electric equipment", not "electric". Always preserve the full noun phrase.
        """

        messages = [
            {"role": "system", "content": system_instruction}
        ]
        
        # Add history so the LLM knows what it already found
        # (History is already formatted as a list of dicts in app.py, so we extend)
        messages.extend(history)
        
        # Add current message
        messages.append({"role": "user", "content": user_message})

        print(f"[LLM-DEBUG] Sending to OpenAI: {user_message}")
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=SPATIALLM_TOOL,   
                tool_choice="auto" 
            )

            message = response.choices[0].message

            if message.tool_calls:
                print("[LLM-DEBUG] OpenAI decided to call TOOLS.")
                actions = []
                for tool in message.tool_calls:
                    if tool.function.name == "detect_3d_objects":
                        args = json.loads(tool.function.arguments)
                        print(f"[LLM-DEBUG] Tool Args: {args}")
                        actions.append({
                            "categories": args.get("categories", []),
                            "detect_type": args.get("detect_type", "object")
                        })
                
                return {
                    "type": "tool_list",
                    "actions": actions
                }
            
            # If no tool is called, it means the LLM found the answer in history
            print(f"[LLM-DEBUG] OpenAI decided to TEXT: {message.content}")
            return {
                "type": "text", 
                "content": message.content
            }
            
        except Exception as e:
            print(f"[LLM-DEBUG] OpenAI Error: {e}")
            return {
                "type": "text", 
                "content": f"Error contacting OpenAI: {e}"
            }