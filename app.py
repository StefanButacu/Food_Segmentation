import json
from functools import wraps

import albumentations as A
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from flask import Flask, request, jsonify
from flask_cors import CORS
from skimage import color
from torch import nn

from model.architectures.SAM_Architecture import SAM_Architecture
from model.architectures.UnetRestNet50 import UNetResNet152
from model.architectures.unet import Unet_model
from model.checkpoints import load_checkpoint
from service.Service import Service
from utils.category_reader import read_categories
from utils.image_byte_converter import convert_byte_int

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
API_KEY = 'my-secret'


CATEGORY_DICT_FILE = 'data/category_id.txt'
category_dict = read_categories(CATEGORY_DICT_FILE)

MODEL_UNET_FILE = 'model/checkpoint.pth.tar'
MODEL_UNET_CE_IOU_LOSS_FILE = 'model/checkpoint-ce_dice_loss.pth.tar'
MODEL_RESNET_FILE = 'model/checkpoint-pretrain.pth.tar'
# MODEL_SAM_FILE = 'model/checkpoint-sam-ce_dice_loss.pth.tar'
MODEL_SAM_FILE = 'model/checkpoint-sam-old.pth.tar'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model = Unet_model().to(DEVICE)
# model = UNetResNet152(104).to(DEVICE)
model = SAM_Architecture(104).to(DEVICE)


# load_checkpoint(torch.load(MODEL_UNET_FILE), model)
# load_checkpoint(torch.load(MODEL_UNET_CE_IOU_LOSS_FILE), model)
# load_checkpoint(torch.load(MODEL_RESNET_FILE), model)
load_checkpoint(torch.load(MODEL_SAM_FILE), model)
#
service = Service()
t1 = A.Compose([
    A.Resize(256, 256),
    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])
path_to_save_image = "E:\Licenta_DOC\App\src\main\\resources\image\\target_python.jpg"


def require_api_key(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-Api-Key') and request.headers.get('X-Api-Key') == API_KEY:
            return view_function(*args, **kwargs)
        else:
            return jsonify({"message":"Unauthorized"}), 403
    return decorated_function

@app.route('/category/<id>', methods=['GET'])
@require_api_key
def get_category_name(id):
    global category_dict
    if id in category_dict:
        return jsonify({'category': category_dict[id]}), 200
    else:
        return jsonify({'error': 'Category not found'}), 404


@app.route('/category', methods=['GET'])
@require_api_key
def get_categories():
    global category_dict
    return category_dict, 200

@app.route('/image', methods=['POST'])
@require_api_key
def get_image_prediction():
    unsigned_bytes = extract_image_bytes(request)
    store_image(path_to_save_image, unsigned_bytes)
    image = Image.open('%s' % path_to_save_image).resize((256,256))
    preds = generate_prediction(image)
    image_with_tranform = np.array(image)
    masked_image, color_map = service.get_overlay_with_map(image_with_tranform, preds)
    data_str_keys = {str(key): value for key, value in color_map.items()}
    json_data = json.dumps(data_str_keys)
    return jsonify(
        {"mask": masked_image,
        "color_map": json_data}), 200


def generate_prediction(image):
    x = np.array(image)
    x = t1(image=x)['image']
    x = x.to(DEVICE)
    x = x.unsqueeze(0)
    softmax = nn.Softmax(dim=1)
    preds = torch.argmax(softmax(model(x)), axis=1).to('cpu')
    return np.array(preds[0, :, :])


def store_image(path_to_save_image, unsigned_bytes):
    with open("%s" % path_to_save_image, "wb") as binary_file:
        for byte1 in unsigned_bytes:
            binary_file.write(byte1.to_bytes(1, byteorder='big'))


def extract_image_bytes(request):
    data = request.json['content']
    if len(data) > 2:
        data = data[1:-1]  # remove square brackets
        bytes = [int(byte) for byte in data.split(',')]
        unsigned_bytes = [convert_byte_int(byte) for byte in bytes]
        return unsigned_bytes
    return []

if __name__ == '__main__':
    app.run()
