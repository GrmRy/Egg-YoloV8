import os

folder_path = 'C:/Users/LENOVO/Documents/Project/Egg-YoloV8/negatif'

for image in os.listdir(folder_path):
    filename = image.split(".")[0]
    print(filename)
    with open(f"{filename}.txt", 'w') as file:
        pass