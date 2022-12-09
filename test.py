import torch
from pprint import pprint
from ModelMaker import MainModel, read_json, get_seen_set

j = read_json("./FakeJson2.json")
seen_set = get_seen_set(j,
                        ["8", "9"],
                        [j["8"], j["9"]])

pprint(seen_set)

my_model = MainModel(j, seen_set)

print(my_model)

data = {
    "0": torch.ones(size=(1, 50)),
    "2": torch.ones(size=(1, 40))
}

print(my_model(data))
