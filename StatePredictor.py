from inference_sdk import InferenceHTTPClient
import base64
from PIL import Image
from io import BytesIO
from ClashRoyalData import TroopSide, ElixirDecode, ElixirCost

API_URL = "http://localhost:9001"
API_KEY = "obQog4mAaBRuPZZBIoti"

WORKSPACE = "clashroyalbot-z9idj"
WORKFLOW  = "detect-count-and-visualize"
IMG_PATH  = r'C:\Users\SlayerDz\Desktop\Screenshot_2025.09.14_21.27.07.354.png'

client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

result = client.run_workflow(
    workspace_name=WORKSPACE,
    workflow_id=WORKFLOW,
    images={"image": IMG_PATH}
)


imgbase = result[0]['img output']

img = base64.b64decode(imgbase)
img = Image.open(BytesIO(img))
img.show()

x = result[0].pop('img output')

def ExtractData(result):
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
    return Slots, Troops, Towers, elixir

Slots, Troops, Towers, Elixir = ExtractData(result)


















