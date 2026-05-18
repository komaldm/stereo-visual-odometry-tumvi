import numpy as np


def calculate_thesis_metrics(gt_file, log_file, seed_file, start_end_drift=0.3269):
    # Initialize variables to handle potential file errors gracefully
    total_path_length_str = "N/A"
    drift_percentage_str = "N/A"
    mttr_str = "N/A"
    max_recovery_str = "N/A"

    peak_seeds_str = "N/A"
    min_seeds_str = "N/A"
    mean_seeds_str = "N/A"
    std_seeds_str = "N/A"

    # 1. Calculate Total Path Length & Drift Percentage
    try:
        gt_data = np.loadtxt(gt_file, delimiter=',')
        positions = gt_data[:, 1:4]  # Extract just x, y, z

        diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        total_path_length = np.sum(distances)
        drift_percentage = (start_end_drift / total_path_length) * 100

        total_path_length_str = f"{total_path_length:.2f} m"
        drift_percentage_str = f"{drift_percentage:.2f} %"
    except Exception:
        total_path_length_str = "ERROR"
        drift_percentage_str = "ERROR"

    # 2. Calculate MTTR (Mean Time to Recovery) from Log
    try:
        with open(log_file, 'r') as f:
            log_lines = f.readlines()

        recovery_times = []
        lost_frame = -1

        for i, line in enumerate(log_lines):
            if "LOST_TRACKING_POINTS" in line or "LOST_PNP_FAILED" in line:
                if lost_frame == -1:
                    lost_frame = i
            elif "EPI_SUCCESS" in line:
                if lost_frame != -1:
                    recovery_times.append(i - lost_frame)
                    lost_frame = -1

        if recovery_times:
            mttr_str = f"{np.mean(recovery_times):.1f} frames"
            max_recovery_str = f"{np.max(recovery_times)} frames"
    except Exception:
        mttr_str = "ERROR"
        max_recovery_str = "ERROR"

    # 3. Calculate Seed Statistics
    try:
        # Load the two-column seed file (timestamp, count)
        seed_data = np.loadtxt(seed_file)
        if seed_data.ndim == 2 and seed_data.shape[1] >= 2:
            counts = seed_data[:, 1]
        else:
            counts = seed_data  # Fallback if it's just a 1D array of counts

        peak_seeds_str = f"{int(np.max(counts))}"
        min_seeds_str = f"{int(np.min(counts))}"
        mean_seeds_str = f"{int(np.mean(counts))}"
        std_seeds_str = f"{np.std(counts):.1f}"
    except Exception:
        peak_seeds_str = "ERROR"
        min_seeds_str = "ERROR"
        mean_seeds_str = "ERROR"
        std_seeds_str = "ERROR"

    start_end_drift_str = f"{start_end_drift:.4f} m"

    # --- TERMINAL TABLE OUTPUT ---
    print("+" + "-" * 52 + "+")
    print("|" + " SYSTEM EVALUATION METRICS ".center(52) + "|")
    print("+" + "-" * 32 + "+" + "-" * 19 + "+")
    print(f"| {'Evaluation Metric':<30} | {'Computed Value':<17} |")
    print("+" + "-" * 32 + "+" + "-" * 19 + "+")
    print(f"| {'Total Path Length':<30} | {total_path_length_str:<17} |")
    print(f"| {'Absolute Start-End Drift':<30} | {start_end_drift_str:<17} |")
    print(f"| {'Drift Percentage':<30} | {drift_percentage_str:<17} |")
    print(f"| {'Mean Time To Recovery (MTTR)':<30} | {mttr_str:<17} |")
    print(f"| {'Max Time Blind':<30} | {max_recovery_str:<17} |")
    print("+" + "-" * 32 + "+" + "-" * 19 + "+")
    print(f"| {'Peak Seed Count':<30} | {peak_seeds_str:<17} |")
    print(f"| {'Minimum Seed Count':<30} | {min_seeds_str:<17} |")
    print(f"| {'Mean Seed Count':<30} | {mean_seeds_str:<17} |")
    print(f"| {'Seed Standard Deviation':<30} | {std_seeds_str:<17} |")
    print("+" + "-" * 32 + "+" + "-" * 19 + "+")


if __name__ == "__main__":
    GROUND_TRUTH_FILE = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_projectR\dataset\dataset-room2_512_16\mav0\mocap0\data.csv"
   # RACKING_LOG_FILE = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_projectR\Results_Room2_Dataset\monocular_keyframes.txt"
   # SEED_LOG_FILE = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_projectR\Results_Room2_Dataset\monocular_seeds.txt"  # Point this to your new seed log

    TRACKING_LOG_FILE = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_projectR\Results_Room2_Dataset\stereo_keyframes.txt"
    SEED_LOG_FILE = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_projectR\Results_Room2_Dataset\stereo_seeds.txt"  # Point this to your new seed log

    # Pass the drift you got from your stereo evaluation script
    calculate_thesis_metrics(GROUND_TRUTH_FILE, TRACKING_LOG_FILE, SEED_LOG_FILE, start_end_drift=0.3269)