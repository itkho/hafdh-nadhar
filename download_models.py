import os

import requests

# URLs to the files
path_file_urls_dict = {
    "src/models/person_prediction": [
        "https://pjreddie.com/media/files/yolov3.weights",
        "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg",
    ],
    "src/models/person_prediction": [
        "https://pjreddie.com/media/files/yolov3.weights",
        "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg",
    ],
}

# Create the folder if it doesn't exist

for folder_path, file_urls in path_file_urls_dict:
    os.makedirs(folder_path, exist_ok=True)

    # Download and save files
    for url in file_urls:
        # Extract the filename from the URL
        filename = os.path.join(folder_path, os.path.basename(url))

        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Save the content to the destination file
            with open(filename, "wb") as file:
                file.write(response.content)
            print(f"Downloaded and saved: {filename}")
        else:
            print(f"Failed to download: {url}")
