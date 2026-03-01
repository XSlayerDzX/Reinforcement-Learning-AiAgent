from inference_sdk import InferenceHTTPClient

API_URL = "http://localhost:9001"
API_KEY = "obQog4mAaBRuPZZBIoti"

WORKSPACE = "clashroyalbot-z9idj"
WORKFLOW  = "detect-and-classify"
IMG_PATH  = r"C:\Users\SK-TECH\Downloads\photo_2026-02-02_17-35-32.jpg"

client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

def predict(image_path):
    result = client.run_workflow(
        workspace_name=WORKSPACE,
        workflow_id=WORKFLOW,
        images={"image": image_path}
    )
    return result



def ExtractSlots(IMG_PATH):
    result = predict(IMG_PATH)
    Slots = {} # slot_1 = "archers",slot_2 = "archers"
    for key, value in result[0].items():
        if "slot_" in key:
            if value["predictions"]:
                card = value["predictions"][0]["class"]
                n = key.split("_")[1]
                Slots.update({f"slot_{n}": card})
    return Slots

if __name__ == "__main__":
    Slots = ExtractSlots(IMG_PATH)
    print("Slots:", Slots)