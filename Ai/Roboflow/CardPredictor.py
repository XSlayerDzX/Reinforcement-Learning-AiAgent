from inference_sdk import InferenceHTTPClient
import base64
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
from Ai.ClashRoyalData import TroopSide, ElixirDecode, ElixirCost

# Load environment variables from .env file
load_dotenv()

API_URL = os.getenv("ROBOFLOW_API_URL", "http://localhost:9001")
API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "clashroyalbot-z9idj")
WORKFLOW = os.getenv("ROBOFLOW_WORKFLOW_CARD", "detect-and-classify")

if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY not found in environment variables. Please check your .env file.")

IMG_PATH = r"C:\Users\abdoa\OneDrive\Desktop\photo_2026-02-26_15-32-03.jpg"

client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

def predict(image_path):
    result = client.run_workflow(
        workspace_name=WORKSPACE,
        workflow_id=WORKFLOW,
        images={"image": image_path}
    )
    return result



def ExtractSlot(imgpath):
    result = predict(imgpath)
    Slots = {} # slot_1 = "archers",slot_2 = "archers"
    for key, value in result[0].items():
        if "slot_" in key:
            if value["predictions"]:
                card = value["predictions"][0]["class"]
                n = key.split("_")[1]
                Slots.update({f"slot_{n}": card})
    return Slots

# if __name__ == "__main__":
#     Slots = ExtractSlot(IMG_PATH)
#     print("Slots:", Slots)