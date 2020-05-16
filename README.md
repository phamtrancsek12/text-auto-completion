TEST
# Text Auto Completion demo with GPT-2 model

This repo contains the source code of python API and HTML demo for Text Auto Completion.

## Model Training
You can use the pretrained GPT-2 model for general text, or follow the instruction at [GPT-2 repo](https://github.com/nshepperd/gpt-2/) to finetune the GPT-2 model on your own data set.

Model in Tensorflow format can be converted to Pytorch using [HuggingFace's Transformers library](https://github.com/huggingface/transformers) 

```python 
from transformers.convert_gpt2_original_tf_checkpoint_to_pytorch import convert_gpt2_checkpoint_to_pytorch

gpt2_checkpoint_path = "path to TF checkpoint"
gpt2_config_file = "path to model config file"
pytorch_dump_folder_path = "path to save pytorch checkpoint"

convert_gpt2_checkpoint_to_pytorch(gpt2_checkpoint_path, gpt2_config_file, pytorch_dump_folder_path)

```

## Text Generation API
Source code of API is in `webapp/src` folder. Tested with `python==3.6` and `transformers==2.4.1`
- Install libraries by `pip install -r requirements.txt`
- Update `MODEL_PATH` in `config.py` with the Python checkpoint path
- Run `python app.py`

The A Flask app is started at port 5000 which allow GET request.
```python
# Request Arguments
input = flask.request.args.get('input')
temperature = float(flask.request.args.get('temperature', default=0.8))
top_p = float(flask.request.args.get('top_p',  default=0.9))
```
 
Please refer to [this article](https://huggingface.co/blog/how-to-generate) for more information about `temperature` and `top_p` values

## HTML demo
Source code of HTML demo is in `webapp/html` folder. Update `URL` constant in `index.js` file by the Text Generation API
