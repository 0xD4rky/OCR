import os

# Define the path to the directory containing the images
directory_path = r'C:\Users\DELL\Amazon\data\train'  # Replace this with the actual path to your images

# List all files ending with '.jpg', removing the '.jpg' extension
image_names = [file[:-4] for file in os.listdir(directory_path) if file.endswith('.jpg')]

# Save the list of image names to a text file
output_file_path = 'image_names.txt'
with open(output_file_path, 'w') as file:
    for name in image_names:
        file.write(name + '\n')

print(f"The list of image names has been saved to {output_file_path}")
