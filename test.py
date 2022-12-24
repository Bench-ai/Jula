import json
import os
from dotenv import load_dotenv
import requests
from pathlib import Path

dotenv_path = Path('D:/BenchAi/Jula/.env')
load_dotenv(dotenv_path=dotenv_path)



url = "http://127.0.0.1:8000/db/insert-layer"
file = "D:\BenchAi\Jula\AdditionalLayers\Linear.py"

j = {
    "layer_name": "TestLinearLayer",
    "layer_parameters": {
        "parameter_list": {
            "in_features": {
                "description": "the size of the input features",
                "default": None,
                "type": "INTEGER",
                "options": None
            },
            "out_features": {
                "description": "the size of the output features",
                "default": None,
                "type": "INTEGER",
                "options": None
            },
            "bias": {
                "description": "Determines whether the layer will learn an additive bias.",
                "default": True,
                "type": "BOOL",
                "options": None
            }
        },
        "input_shape": [
            {
                "name": "input",
                "shape": [
                    [
                        "{}",
                        [
                            "BATCH"
                        ]
                    ],
                    [
                        "{}",
                        [
                            "in_features"
                        ]
                    ]
                ]
            }
        ],
        "output_shape": [
            {
                "name": "output",
                "shape": [
                    [
                        "{}",
                        [
                            "BATCH"
                        ]
                    ],
                    [
                        "{}",
                        [
                            "out_features"
                        ]
                    ]
                ]
            }
        ]
    },
    "tags": [
        [
            "LAYER",
            "LinearLayer"
        ],
        [
            "LAYER",
            "FullyConnected"
        ],
        [
            "LAYER",
            "DENSELAYER"
        ]
    ]
}

payload = {"param_1": "value_1", "param_2": "value_2"}
files = {
    'json': (None, json.dumps(j), 'application/json'),
    'file': (os.path.basename(file), open(file, 'rb'), 'application/octet-stream')
}

resp = requests.post(url,
                     files=files,
                     headers={"Authorization": "Api-Key {}".format(os.getenv("KEY"))})

