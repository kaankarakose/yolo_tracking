import subprocess
import os
from pathlib import Path
def run_tracking(input_folder, output_folder, yolo_model='yolox_x', tracking_method='botsort', reid_model='clip_market1501.pt', device='0'):
    """
    Run object tracking on video files in the input folder and save results in the output folder.

    :param input_folder: Path to the folder containing video files.
    :param output_folder: Path to the folder where output will be saved.
    :param yolo_model: Name of the YOLO model to use.
    :param tracking_method: Tracking method to use.
    :param reid_model: Path to the reID model.
    :param device: Device to run the tracking on.
    """
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input directory
    for video_file in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_file)
        
        # Construct the output path for this video
        #video_output_folder = os.path.join(output_folder, os.path.splitext(video_file)[0])

        

        # Construct the command
        command = [
            'python3', './examples/track.py',
            '--yolo-model', yolo_model,
            '--tracking-method', tracking_method,
            '--source', video_path,
            '--reid-model', reid_model,
            '--device', device,
            '--save',
            '--save-mot',
            '--device','1',
            '--project', output_folder,
            '--name',os.path.splitext(video_file)[0],
            '--save-txt',
        ]


        # Run the command
        subprocess.run(command)

if __name__ == "__main__":
    input_folder = '/nas/project_data/B1_Behavior/rush/object-manipulation/make_believe/dataset'  # Update this path to your input videos folder
    output_folder = '/nas/project_data/B1_Behavior/rush/object-manipulation/make_believe/table'  # Update this path to your desired output folder
    run_tracking(input_folder, output_folder)
