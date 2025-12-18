import os

def save_image_paths_to_txt(folder_path, output_file):
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff",  ".PNG"}

    with open(output_file, "w") as file:
        for root, _, files in os.walk(folder_path):
            sorted_files = sorted(files)
            for filename in sorted_files:
                if os.path.splitext(filename)[1].lower() in image_extensions:
                    full_path = os.path.abspath(os.path.join(root, filename))
                    file.write(full_path.replace("\\", "/") + "\n")

folder_path = "path"  
output_file = "path"  
save_image_paths_to_txt(folder_path, output_file)