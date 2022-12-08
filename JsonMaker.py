import json
import uuid


def construct_json(layer_list):
    with open("test.py", "w") as file:
        json.dump(layer_list, file)


if __name__ == '__main__':
    uuid_list = [str(uuid.uuid4()) for _ in range(5)]

    layer_data = [
        {
            "layer_id": uuid_list[0],
            "layer_type": {
                "name": "InputLayer",
                "target_shape": [1000]
            },
            "input_layer": None,
            "output_layer": [uuid_list[0]],
        },
        {
            "layer_id": uuid_list[1],
            "layer_type": {
                "name": "LinearLayer",
                "target_shape": [1000],
                "out_features": 500,
                "bias": True
            },
            "input_layer": [uuid_list[0]],
            "output_layer": [uuid_list[2]],
        },
        {
            "layer_id": uuid_list[2],
            "layer_type": {
                "name": "LinearLayer",
                "target_shape": [500],
                "out_features": 100,
                "bias": True
            },
            "input_layer": [uuid_list[1]],
            "output_layer": [uuid_list[3]],
        },
        {
            "layer_id": uuid_list[3],
            "layer_type": {
                "name": "LinearLayer",
                "target_shape": [100],
                "out_features": 1,
                "bias": True
            },
            "input_layer": [uuid_list[2]],
            "output_layer": [uuid_list[4]],
        },
        {
            "layer_id": uuid_list[4],
            "layer_type": {
                "name": "SigmoidLayer",
                "target_shape": [1],
            },
            "input_layer": [uuid_list[3]],
            "output_layer": None,
        }
    ]

    construct_json(layer_data)
