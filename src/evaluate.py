import torch
from torchvision import transforms
from PIL import Image
from net import Model

def load_model(model_path):
    model = Model()
    # Load the entire checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Check if 'state_dict' key exists in the checkpoint
    if 'state_dict' in checkpoint:
        # Load state dictionary specifically
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Directly load the state dictionary (assuming the file only contains the state dictionary)
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)

def evaluate_image(image_path, model):
    image = preprocess_image(image_path)
    with torch.no_grad():
        prediction = model(image)
        # Interpretation of prediction depends on your model's output
        return prediction

if __name__ == "__main__":
    model = load_model('saved_models/model.ckpt')
    image_path = 'data/evalData/0.jpg'
    prediction = evaluate_image(image_path, model)
    print(prediction)
