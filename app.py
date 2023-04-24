import json

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
from model.checkpoints.checkpoints import load_checkpoint
from service.Service import Service
from utils.category_reader import read_categories, aggregate_category_freq
from utils.image_byte_converter import convert_byte_int

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

CATEGORY_DICT_FILE = 'data/category_id.txt'
category_dict = read_categories(CATEGORY_DICT_FILE)

MODEL_PARAMETERS_FILE = 'model/checkpoints/checkpoint.pth.tar'
MODEL_PRETRAINED_FILE = 'model/checkpoints/checkpoint-pretrain.pth.tar'
MODEL_PARAMETERS_LOSS_FILE = 'model/checkpoints/checkpoint-ce_dice_loss.pth.tar'
MODEL_SAM_FILE = 'model/checkpoints/checkpoint-sam-ce_dice_loss.pth.tar'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model = Unet_model().to(DEVICE)
# model = UNetResNet152(104).to(DEVICE)
model = SAM_Architecture(104).to(DEVICE);
load_checkpoint(torch.load(MODEL_SAM_FILE), model)
# load_checkpoint(torch.load(MODEL_PARAMETERS_LOSS_FILE), model)
service = Service()


@app.route('/category/<id>', methods=['GET'])
def get_category_name(id):
    global category_dict
    if id in category_dict:
        return jsonify({'category': category_dict[id]}), 200
    else:
        return jsonify({'error': 'Category not found'}), 404


@app.route('/category', methods=['GET'])
def get_categories():
    global category_dict
    return category_dict, 200


@app.route('/overlay-image', methods=['GET'])
def get_overlay_image():
    photo = np.array(Image.open('E:\Licenta_DOC\App\src\main\\resources\image\\target_python.jpg'))
    mask = np.random.randint(1, 4, (photo.shape[0], photo.shape[1]))
    mask[:300, :] = 1
    mask[300:, ] = 0
    overlay_photo = color.label2rgb(mask, photo, saturation=1, alpha=0.5, bg_color=None)

    overlay_photo = np.rot90(overlay_photo)
    overlay_photo = np.rot90(overlay_photo)
    overlay_photo = np.rot90(overlay_photo)
    overlay_photo = np.fliplr(overlay_photo)
    Image.fromarray((overlay_photo * 255).astype(np.uint8)).save('overlay-photo.png')
    return jsonify({'mask': (overlay_photo * 255).astype(np.uint8).tolist()}), 200


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/image', methods=['POST'])
def get_image_content():  # put application's code here
    unsigned_bytes = extract_image_bytes(request)
    #######
    path_to_save_image = "E:\Licenta_DOC\App\src\main\\resources\image\\target_python.jpg"
    with open("%s" % path_to_save_image, "wb") as binary_file:
        for byte1 in unsigned_bytes:
            binary_file.write(byte1.to_bytes(1, byteorder='big'))
    #######
    image = Image.open('%s' % path_to_save_image)
    x = np.array(image)
    image = image.resize((256,256))
    image_with_tranform = np.array(image)
    t1 = A.Compose([
        A.Resize(256, 256),
        A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    x = t1(image=x)['image']
    x = x.to(DEVICE)
    x = x.unsqueeze(0)
    softmax = nn.Softmax(dim=1)
    preds = torch.argmax(softmax(model(x)), axis=1).to('cpu')
    preds1 = np.array(preds[0, :, :])

    # print(aggregate_category_freq(category_dict, preds1))
    ### play with the pixel values

    # masked_image, color_map = service.get_dummy_overlay_with_map()
    masked_image, color_map = service.get_overlay_with_map(image_with_tranform, preds1)
    data_str_keys = {str(key): value for key, value in color_map.items()}
    json_data = json.dumps(data_str_keys)


    return jsonify(
        {"mask": masked_image,
        "color_map": json_data}
    ), 200

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
