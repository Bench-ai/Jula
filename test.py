import torch
from pprint import pprint

from torch import nn

from AdditionalLayers.BaseLayer import InputLayer
from AdditionalLayers.Linear import LinearLayer
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


class TestLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.__fc1 = nn.Linear(10,
                               30)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        return self.__fc1(x)


model = TestLayer()

print(model(torch.ones(size=(1, 10))))
