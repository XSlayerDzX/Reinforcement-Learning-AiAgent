#importthe screenshot module
#-->
#img_path = r"C:\Users\SlayerDz\Desktop\Screenshot_2025.09.14_21.27.07.354.png"
from StatePredictor import ExtractData
import State_Tracker

def ExtractSlots(Slots):
    slot1 = Slots.get("slot_1")  # None if missing
    slot2 = Slots.get("slot_2")
    slot3 = Slots.get("slot_3")
    slot4 = Slots.get("slot_4")
    return slot1, slot2, slot3, slot4


def ExtractTower(Towers, side, tower_type):
    if tower_type in Towers:
        return 1
    else:
        return 0

def ExtractElixir(Elixir):
    if not Elixir:
        return 0
    else:
        State_Tracker.CurrentElixir = Elixir
        return Elixir

def ExtractCard(Troops, card_name):
    if card_name in Troops:
        print(Troops[card_name])
        position= Troops[card_name][0]
        x, y = position
        return 1, (round(int(x),2)), round(int(y),2)
    else:
        return 0, None, None

def ExtractDistance(Troops_ally,Troops_enemy, ally_card, enemy_card):
    if ally_card in Troops_ally and enemy_card in Troops_enemy:
        position_ally= Troops_ally[ally_card][0]
        position_enemy= Troops_enemy[enemy_card][0]
        x1, y1 = position_ally
        x2, y2 = position_enemy
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return round(int(distance),2)
    else:
        return 10000000


def Output_Dataset_Schema(action, pos_x, pos_y,id):
    """This function defines the schema for the dataset row. It takes the action and the position (x, y) as input
    and returns a dictionary representing the dataset output."""
    schema = {
        "id": id,
        "action": action,
        "pos_x": pos_x,
        "pos_y": pos_y,
    }
    return schema




def Create_Dataset_Row(imgpath,id,match_id):
    if not imgpath:
        print("No image path provided.")
        return None

    try:
        data = ExtractData(imgpath)
        if data is None:
            print("Data extraction failed.")
            return None

        slots, troops_ally, troops_enemy, towers, elixir = ExtractData(imgpath)
        print(slots)
    except Exception as e:
        print(f"Error extracting data from image: {e}")
        return None

    # --- Slots ---
    slot_1, slot_2, slot_3, slot_4 = ExtractSlots(slots)

    # --- Elixir ---
    elixir = ExtractElixir(elixir)

    # --- Towers (ally) ---
    ally_prince_tower_left = ExtractTower(towers, "ally", "left_princess_tower")
    ally_prince_tower_right = ExtractTower(towers, "ally", "right_princess_tower")
    ally_king_tower = ExtractTower(towers, "ally", "king_tower")

    # --- Towers (enemy) ---
    enemy_prince_tower_left = ExtractTower(towers, "enemy", "enemy_left_princess_tower")
    enemy_prince_tower_right = ExtractTower(towers, "enemy", "enemy_right_princess_tower")
    enemy_king_tower = ExtractTower(towers, "enemy", "enemy_king_tower")

    # --- Card list ---
    cards = [
        "archers",
        "giant",
        "minions",
        "goblin cage",
        "goblin gang",
        "goblin hut",
        "goblins",
        "knight",
        "mini pekka",
        "musketeer",
        "spear goblins",
    ]

    # --- Card features: presence + x + y (ally & enemy) ---
    card_features = {}
    for card in cards:
        # Ally
        ally_present, ally_x, ally_y = ExtractCard(troops_ally, card)
        card_features[f"{card}_ally"] = ally_present
        card_features[f"{card}_ally_x"] = ally_x
        card_features[f"{card}_ally_y"] = ally_y

        # Enemy
        enemy_card = f"enemy_{card}"
        enemy_present, enemy_x, enemy_y = ExtractCard(troops_enemy, enemy_card)
        card_features[f"{card}_enemy"] = enemy_present
        card_features[f"{card}_enemy_x"] = enemy_x
        card_features[f"{card}_enemy_y"] = enemy_y

    # --- Distance features (ally → enemy) ---
    distance_features = {}
    for ally_card in cards:
        for enemy_card in cards:
            d = ExtractDistance(troops_ally, troops_enemy, ally_card, f"enemy_{enemy_card}")
            key = f"{ally_card}_ally_enemy_{enemy_card}_d"
            distance_features[key] = d

    # --- Assemble full feature dict ---
    feature_dict = {
        "match_id": match_id,
        "id": id,
        "slot_1": slot_1,
        "slot_2": slot_2,
        "slot_3": slot_3,
        "slot_4": slot_4,
        "Elixir": elixir,
        "ally_prince_tower_left": ally_prince_tower_left,
        "ally_prince_tower_right": ally_prince_tower_right,
        "ally_king_tower": ally_king_tower,
        "enemy_prince_tower_left": enemy_prince_tower_left,
        "enemy_prince_tower_right": enemy_prince_tower_right,
        "enemy_king_tower": enemy_king_tower,
    }

    # Merge card features
    feature_dict.update(card_features)

    # Merge distance features
    feature_dict.update(distance_features)

    return feature_dict

#dict= Create_Dataset_Row(imgpath=img_path)
#print(dict)