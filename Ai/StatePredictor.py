from inference_sdk import InferenceHTTPClient
import base64
from PIL import Image
from io import BytesIO
from ClashRoyalData import TroopSide, Tower_Side , ElixirDecode, ElixirCost

API_URL = "http://localhost:9001"
API_KEY = "obQog4mAaBRuPZZBIoti"

WORKSPACE = "clashroyalbot-z9idj"
WORKFLOW  = "detect-count-and-visualize"
IMG_PATH  = r"C:\Users\SlayerDz\Desktop\Screenshot_2025.09.14_21.27.07.354.png"

client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

def predict(image_path):
    result = client.run_workflow(
        workspace_name=WORKSPACE,
        workflow_id=WORKFLOW,
        images={"image": image_path}
    )
    return result

def Show_img(result):
    imgbase = result[0]['img output']
    img = base64.b64decode(imgbase)
    img = Image.open(BytesIO(img))
    img.show()


def ExtractData(imgpath):
    result = predict(imgpath)
    Show_img(result)
    result[0].pop('img output')
    Slots = {} # slot_1 = "archers",slot_2 = "archers"
    Troops_ally = {} # "knight" : (x,y), ally
    Troops_enemy = {} # "knight" : (x,y), enemy
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
                    if object["class"] in TroopSide:
                        side = TroopSide[object["class"]]
                        if side == "ally":
                            Elixircost = ElixirCost[object["class"]]
                            position = (object["x"], object["y"])
                            Troops_ally.update({object["class"]: (position, side, Elixircost)})
                        elif side == "enemy":
                            position = (object["x"], object["y"])
                            Troops_enemy.update({object["class"]: (position, side)})
                    elif "tower" in object["class"]:
                        side = Tower_Side[object["class"]]
                        Towers.update({object["class"]: side})
                    elif "elixir" in object["class"]:
                        elixir = ElixirDecode[object["class"]]
    return Slots, Troops_ally, Troops_enemy, Towers, elixir

Slots, Troops_ally, Troops_enemy, Towers, Elixir = ExtractData(imgpath=IMG_PATH)
if __name__ == "__main__":
 print("Slots:", Slots)
 print("Troops:", Troops_ally)
 print("Troops:", Troops_enemy)
 print("Towers:", Towers)
 print("Elixir:", Elixir)

















