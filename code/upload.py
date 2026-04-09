import os

local_path = "D:\Projekt magisterki\spectograms"
remote_path = "Praca magisterska/data/spectograms"

def upload_file_to_google_drive(file_name, sent, max_files):
    os.system(f'rclone copy "{local_path}\{file_name}" "{remote_path}"')
    if(sent%10==0):
        print(f"Wyslano {sent}/{max_files}")




# rclone copy "D:\Projekt magisterki\spectograms" gdrive:"Praca magisterska/data/spectograms"