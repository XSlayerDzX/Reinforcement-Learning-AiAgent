#import the screenshot module
#-->
from StatePredictor import ExtractData







def ExtractSlots(Slots):
    slot1 = Slots["slot_1"]
    slot2 = Slots["slot_2"]
    slot3 = Slots["slot_3"]
    slot4 = Slots["slot_4"]
    return slot1, slot2, slot3, slot4

def ExtractTower(Towers, side, tower_type):
    if tower_type in Towers:
        return 1
    else:
        return 0
    pass

def ExtractElixir(Elixir):
    if not Elixir:
        return None
    else:
        return Elixir

def ExtractCard(Troops, card_name, side):
    if Troops == "ally":
        if card_name in Troops:
            position, _, _ = Troops[card_name]
            x, y = position
            return 1, (round(int(x),2)), round(int(y),2)
        else:
            return 0, None, None
    else:
        if card_name in Troops:
            position, _, _ = Troops[card_name]
            x, y = position
            return 1, (round(int(x),2)), round(int(y),2)
        else:
            return 0, None, None

def ExtractDistance(Troops_ally,Troops_enemy, ally_card, enemy_card):
    pass



def CreateDataSet(imgpath):
    if not imgpath:
        print("No image path provided.")
        return None

    Slots, Troops_ally, Troops_enemy, Towers, Elixir = ExtractData(imgpath)

    # --- Slots ---
    slot_1, slot_2, slot_3, slot_4 = ExtractSlots(Slots)

    # --- Elixir ---
    Elixir = ExtractElixir(Elixir)

    # --- Towers (ally) ---
    ally_prince_tower_left = ExtractTower(Towers, "ally", "left_princess_tower")
    ally_prince_tower_right = ExtractTower(Towers, "ally", "right_princess_tower")
    ally_king_tower = ExtractTower(Towers, "ally", "king_tower")

    # --- Towers (enemy) ---
    enemy_prince_tower_left = ExtractTower(Towers, "enemy", "enemy_left_princess_tower")
    enemy_prince_tower_right = ExtractTower(Towers, "enemy", "enemy_right_princess_tower")
    enemy_king_tower = ExtractTower(Towers, "enemy", "enemy_king_tower")

    # --- Card list ---
    cards = [
        "archers",
        "giant",
        "minions",
        "goblin_cage",
        "goblin_gang",
        "goblin_hut",
        "goblins",
        "knight",
        "mini_pekka",
        "musketeer",
        "spear_goblins",
    ]

    # --- Card features: presence + x + y (ally & enemy) ---
    card_features = {}
    for card in cards:
        # Ally
        ally_present, ally_x, ally_y = ExtractCard(Troops_ally, card)
        card_features[f"{card}_ally"] = ally_present
        card_features[f"{card}_ally_x"] = ally_x
        card_features[f"{card}_ally_y"] = ally_y

        # Enemy
        enemy_present, enemy_x, enemy_y = ExtractCard(Troops_enemy, card, "enemy")
        card_features[f"{card}_enemy"] = enemy_present
        card_features[f"{card}_enemy_x"] = enemy_x
        card_features[f"{card}_enemy_y"] = enemy_y

    # --- Distance features (ally → enemy) ---
    distance_features = {}
    for ally_card in cards:
        for enemy_card in cards:
            d = ExtractDistance(Troops_ally, Troops_enemy, ally_card, enemy_card)
            key = f"{ally_card}_ally_enemy_{enemy_card}_d"
            distance_features[key] = d

    # --- Assemble full feature dict ---
    feature_dict = {
        "slot_1": slot_1,
        "slot_2": slot_2,
        "slot_3": slot_3,
        "slot_4": slot_4,
        "Elixir": Elixir,
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
