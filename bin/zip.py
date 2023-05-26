import zipfile
import os

# Set the path to the folder you want to zip
folder_path = "/media/jorrit/Storage SSD/fathomnet/query2label/q2l_labeller"

# Set the path to the directory where you want to save the zip file
save_path = "/media/jorrit/Storage SSD/fathomnet/zip"

# Set the name of the zip file you want to create
zip_name = "q2l_labeller.zip"

# Create a ZipFile object
zip_file = zipfile.ZipFile(os.path.join(save_path, zip_name), mode="w")

# Iterate over all the files in the folder and add them to the zip file
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        # Get the relative path of the file from the folder_path
        relative_path = os.path.relpath(file_path, folder_path)
        zip_file.write(file_path, arcname=relative_path)

# Close the zip file
zip_file.close()

print(f"Zip file created at {os.path.join(save_path, zip_name)}")
