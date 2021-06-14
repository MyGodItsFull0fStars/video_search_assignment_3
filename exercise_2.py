import torch
from torchvision import models, transforms
from typing import Tuple, List
from PIL import Image

from enum import Enum

class ModelType(Enum):
    ALEX_NET = 0,
    GOOGLE_NET = 1,
    VGG = 2,
    RES_NET = 3


def get_classes() -> list:
    with open('imagenet1000_classes.txt') as image_file:
        ret_val = [line.strip() for line in image_file.readlines()]
    return ret_val


def get_transform() -> transforms:
    """
    [1] -> Transform instance which is a combination of all the image transformations to be carried out on the input image
    [2] -> Rezise the image to 256x256
    [3] -> Crop the image to 224x224 about the center
    [4] -> Convert the image to Pytorch Tensor data type
    [5-7] -> Normalize the image by setting its mean and standard deviation to the specified values
    """
    ret_val = transforms.Compose([  # [1]
        transforms.Resize(256),  # [2]
        transforms.CenterCrop(224),  # [3]
        transforms.ToTensor(),  # [4]
        transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406],  # [6]
            std=[0.229, 0.224, 0.225]  # [7]
        )
    ])
    return ret_val


def get_model(model_type: ModelType = ModelType.ALEX_NET) -> models:
    if model_type == ModelType.ALEX_NET:
        model = models.alexnet(pretrained=True)
    elif model_type == ModelType.GOOGLE_NET:
        model = models.googlenet(pretrained=True)
    elif model_type == ModelType.VGG:
        model = models.vgg11(pretrained=True)
    elif model_type == ModelType.RES_NET:
        model = models.resnet101(pretrained=True)

    model.eval()
    return model


def get_prediction_and_percentage(image_path: str, model: models) -> Tuple[str, float]:
    global classes
    image = Image.open(image_path)
    transformed_image = get_transformed_image(image)
    transformed_batch = torch.unsqueeze(transformed_image, 0)
    model_output = model(transformed_batch)

    _, index = torch.max(model_output, 1)
    percentage = torch.nn.functional.softmax(model_output, dim=1)[0] * 100

    return classes[index], percentage[index].item()


def get_transformed_image(image):
    transform = get_transform()
    transformed_image = transform(image)
    return transformed_image


def get_image_paths() -> List[str]:
    # TODO
    pass



if __name__ == '__main__':
    classes = get_classes()

    alexnet_model = get_model()
    print(get_prediction_and_percentage('testimages/dogs/balu_hat.jpeg', alexnet_model))

    googlenet_model = get_model(ModelType.GOOGLE_NET)
    print(get_prediction_and_percentage('testimages/dogs/balu_hat.jpeg', googlenet_model))

    
    
    
