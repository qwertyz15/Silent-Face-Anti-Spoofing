import os
import base64
import requests

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
    return encoded_string

def main():
    # Path to the folder containing images
    images_folder = "/home/dev/Documents/Silent-Face-Anti-Spoofing/images/dataset_new/b"

    # URL of the Flask server
    server_url = "http://127.0.0.1:5000/predict"

    # Collect base64 encoded images
    base64_images = {}
    for filename in os.listdir(images_folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(images_folder, filename)
            encoded_image = encode_image(image_path)
            base64_images[filename] = encoded_image

    # Send images to the server
    response = requests.post(server_url, json={"images": base64_images})

    # Print the prediction result
    if response.status_code == 200:
        results = response.json()
        print(results)

    else:
        print("Failed to get a valid response from the server.")

if __name__ == "__main__":
    main()
