import matplotlib.pyplot as plt

# Update this path to where your trajectory.txt is located
log_file_path = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_projectC\Results_Room2_Dataset\monocular_trajectory.txt"

frames = []
cumulative_keyframes = []
current_kf_count = 0

# Read the log file# Added the \ right after C:
# log_file_path = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_projectC\Results_Room2_Dataset\monocular_trajectory.txt"
with open(log_file_path, 'r') as f:
    for line_idx, line in enumerate(f):
        parts = line.strip().split()
        if not parts:
            continue

        # ASSUMPTION: Your file has the frame index or timestamp in the first column,
        # and the string 'status' (e.g., NEW_KEYFRAME) in the last column.
        # Change these indices if your format is different!
        try:
            frame_id = line_idx  # Or use float(parts[0]) if you log timestamps
            status = parts[-1]
        except ValueError:
            continue

        # Check if this frame triggered a new keyframe or stereo stabilization
        if "KEYFRAME" in status or "METRIC_STABILIZED_KF" in status:
            current_kf_count += 1

        frames.append(frame_id)
        cumulative_keyframes.append(current_kf_count)

# Plotting the data to match the academic style in your image
plt.figure(figsize=(8, 5), dpi=300)  # High DPI for paper publication

plt.plot(frames, cumulative_keyframes, label="Proposed Stereo VO", color='blue', linewidth=1.5)

# Formatting the plot
plt.title("Evolution of the Number of Keyframes in the Map (Room 2)", fontsize=14, fontweight='bold')
plt.xlabel("Frames", fontsize=12)
plt.ylabel("KeyFrames", fontsize=12)

# Adding the grid to match the ORB-SLAM graph style
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend(loc="lower right", fontsize=12)

# Save the figure directly for your LaTeX document
plt.tight_layout()
plt.savefig("room2_keyframe_evolution.png", bbox_inches='tight')
plt.show()