import subprocess

RCLONE_PATH = r"D:\Programy [Studia]\rclone\rclone.exe"

def upload_folder_to_gdrive(local_folder, remote_disk, remote_path):
    subprocess.run([
        RCLONE_PATH,
        "copy",
        local_folder,
        f"{remote_disk}:{remote_path}",
        "--transfers=8",
        "--checkers=16"
    ])