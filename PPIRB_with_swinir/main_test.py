import subprocess


def run_command(command: str, dataset: str, specific_files: dict):
    """Execute a shell command and print specific lines of the output with the dataset name."""
    try:
        print(f"Executing: {command}")
        process = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)

        # Keep and print the entire output
        output_lines = process.stdout.split('\n')
        print(f"--- Full Output for {dataset} ---")
        for line in output_lines:
            print(line)

        # Print specific results for given dataset and files
        if dataset in specific_files:
            print(f"--- Specific Results for {dataset} ---")
            for line in output_lines:
                for filename in specific_files[dataset]:
                    if filename in line:
                        print(f"{dataset} ({filename}): {line}")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


def generate_command(task: str, scale: int, training_patch_size: int, model_path: str, lr_folder: str, gt_folder: str):
    """Generate the command string for running the model."""
    return f"python main_test_Cotswinir.py --task {task} --scale {scale} --training_patch_size {training_patch_size} --model_path {model_path} --folder_lq {lr_folder} --folder_gt {gt_folder}"


def main():
    # Parameters
    task = "lightweight_sr"
    scale = 4
    training_patch_size = 64  #48  64
    model_name = " "  # Use a variable to store the model name

    model_path = f"E:/{model_name}"

    # Folders
    base_folder = " " #Test set path
    datasets = ["Set5", "Set14", "BSD100", "Urban100", "Manga109"]   #"Set5", "Set14", "BSD100", "Urban100", "Manga109"
    lr_suffix = f"LRbicx{scale}"  # LR folder suffix

    # Specify the specific files for certain datasets

    # Generate and run commands for each dataset
    for dataset in datasets:
        # lr_folder = f"{base_folder}/{dataset}/{lr_suffix}/LR"
        # gt_folder = f"{base_folder}/{dataset}/{lr_suffix}/HR"  # GTmod12 original

        lr_folder = f"{base_folder}/{dataset}/{lr_suffix}"
        gt_folder = f"{base_folder}/{dataset}/original"  # GTmod12 original
        command = generate_command(task, scale, training_patch_size, model_path, lr_folder, gt_folder)
        run_command(command, dataset)


if __name__ == "__main__":
    main()
