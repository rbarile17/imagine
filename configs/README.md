# Configs

Specify the configuration for each step of the pipeline in this directory.

Create a file `<model>_<dataset>.json` for each model and dataset you want to experiment with.

The file is structured as follows:

```json
{
    "model": "<model_name>",
    "model_params": {
        "param_name": "<param_value>"
    },
    "training": {
        "param_name": "<param_value>"
    },
    "explanation": {
        "param_name": "<param_value>"
    },
    "evaluation": {
        "param_name": "<param_value>"
    }
}
```

Set in `model_params` the parameters to initialize the model, such as `dimension` and `init_scale`. In `training` set those required for the training loop.
In `kelpie` and `evaluation` set the same parameter of training, possibly with different values.
