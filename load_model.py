import torch
from src.model_lib.MiniFASNet import MiniFASNetV1  # Replace with the correct model class as needed
from src.anti_spoof_predict import AntiSpoofPredict  # Assuming this class handles the prediction

def load_pretrained_model(model_path, device='cpu'):
    """
    Load a pre-trained model from the specified path.
    """
    # Assuming MiniFASNetV1, but replace with the correct model class
    model = MiniFASNetV1()  # or use MultiFTNet or other model classes as required
    model.to(device)

    # Load the pre-trained weights
    state_dict = torch.load(model_path, map_location=device)

    # Adjust for the DataParallel wrapper
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.` prefix
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()  # Set the model to evaluation mode

    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '/home/dev/Documents/Silent-Face-Anti-Spoofing/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth'  # Update with actual model path

    # Load the model
    model = load_pretrained_model(model_path, device=device)

    # For demonstration, create an AntiSpoofPredict instance for prediction (modify as per your project setup)
    predictor = AntiSpoofPredict(device)
    
    # Dummy input for demonstration (replace with actual input)
    dummy_input = torch.randn(1, 3, 80, 80, device=device)  # Replace with actual input dimensions

    # Perform inference (modify according to how your project handles prediction)
    with torch.no_grad():
        output = predictor.predict(dummy_input, model)

    print("Model output:", output)

if __name__ == "__main__":
    main()

