import os
import argparse
import requests
import base64

def main(image_dir, url):
    # List to store the Base64 encoded images
    base64_image_list = []

    # Iterate over each image file in the directory
    for filename in os.listdir(image_dir):
        # Check if the file is an image (you can add more file extensions as needed)
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Read the image file as binary
            with open(os.path.join(image_dir, filename), 'rb') as f:
                # Encode the image to Base64
                base64_image = base64.b64encode(f.read()).decode('utf-8')

            # Append the Base64 encoded image to the list
            base64_image_list.append(base64_image)

    # Make a POST request to the Flask server with the list of Base64 encoded images
    response = requests.post(url, json={'images': base64_image_list})

    # Print the prediction result
    if response.status_code == 200:
        results = response.json()
        print(results)
    else:
        print("Failed to get a valid response from the server.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and perform anti-spoofing detection.")
    parser.add_argument("--image_dir", type=str, default="/home/dev/Documents/Silent-Face-Anti-Spoofing/images/dataset_new/b", help="Directory containing images")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:5000/predict", help="URL of the Flask server")
    args = parser.parse_args()
    
    main(args.image_dir, args.url)
