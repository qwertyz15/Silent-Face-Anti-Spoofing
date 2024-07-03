import torch
from MiniFASNet import MiniFASNetV1SE  # Import the correct model class

def print_model_weights(model):
    for name, param in model.state_dict().items():
        print(f"Layer: {name}")
        print(f"Values: \n{param}\n") 

def load_pretrained_model(model_path, device='cpu'):
    """
    Load a pre-trained MiniFASNetV1SE model from the specified path.
    """
    # Create an instance of the MiniFASNetV1SE model
    model = MiniFASNetV1SE(embedding_size=128, conv6_kernel=(5, 5), drop_p=0.75, num_classes=3, img_channel=3)
    model.to(device)

    # Load the pre-trained weights
    state_dict = torch.load(model_path, map_location=device)

    # Adjust for the DataParallel wrapper if necessary
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print_model_weights(model)
    model.eval()  # Set the model to evaluation mode

    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '/home/dev/Documents/Silent-Face-Anti-Spoofing/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth'  # Update with actual model path

    # Load the model
    model = load_pretrained_model(model_path, device=device)

    # Add any additional code for testing or inference here
    print("Model loaded successfully.")

if __name__ == "__main__":
    main()
