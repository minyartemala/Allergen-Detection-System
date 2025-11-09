import requests

# Path to your local image
image_path = "baristella rocher.jpeg"  # change to your actual image path

# API endpoint URL
url = "http://127.0.0.1:5000/predict"

# Open the image file in binary mode and send it as multipart/form-data
with open(image_path, 'rb') as image_file:
    files = {'image': (image_path, image_file, 'image/jpeg')}
    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            print("✅ Response from server:")
            print(response.json())
        else:
            print(f"❌ Failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
