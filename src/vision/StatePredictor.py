from inference_sdk import InferenceHTTPClient
from src.training.ClashRoyalData import TroopSide, ElixirDecode, ElixirCost

API_URL = "http://localhost:9001"
API_KEY = "obQog4mAaBRuPZZBIoti"

WORKSPACE = "clashroyalbot-z9idj"
WORKFLOW  = "detect-count-and-visualize"
IMG_PATH  = r"C:\Users\SK-TECH\Downloads\photo_2026-02-02_17-35-32.jpg"

client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)
def run_inference(img_path=IMG_PATH):
    result = client.run_workflow(
    workspace_name=WORKSPACE,
    workflow_id=WORKFLOW,
    images={"image": IMG_PATH}
    )
    return result

#imgbase = result[0]['img output']

#img = base64.b64decode(imgbase)
#img = Image.open(BytesIO(img))
#img.show()

#x = result[0].pop('img output')
def ExtractData(imsg_path=IMG_PATH):
    result = run_inference()
    Slots = {} # slot_1 = "archers",slot_2 = "archers"
    Troops = {} # "knight" : (x,y), ally
    Towers = {} # "left_princess_tower" : ally
    elixir = 0
    for key, value in result[0].items():
        if "slot_" in key:
            if value["predictions"]:
                card = value["predictions"][0]["class"]
                n = key.split("_")[1]
                Slots.update({f"slot_{n}": card})
        elif key == "predictions":
            if value["predictions"]:
                for object in value["predictions"]:
                    if object["class"] in ElixirCost:
                        Elixircost = ElixirCost[object["class"]]
                        position = (object["x"], object["y"])
                        side = TroopSide[object["class"]]
                        Troops.update({object["class"]: (position, side, Elixircost)})
                    elif "tower" in object["class"]:
                        side = TroopSide[object["class"]]
                        Towers.update({object["class"]: side})
                    elif "elixir" in object["class"]:
                        elixir = ElixirDecode[object["class"]]
    return Slots, Troops, elixir , Towers

Slots, Troops, Elixir,Towers  = ExtractData(IMG_PATH)
if __name__ == "__main__":
 print("Slots:", Slots)
 print("Troops:", Troops)
 print("Towers:", Towers)
 print("Elixir:", Elixir)

















