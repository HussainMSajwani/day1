# Neuromorphic Vision Workshop: Day 1

## Data collection and preprocessing

Welcome to Day 1 of the DroneLeaf Neuromorphic Vision Workshop!

After collecting data during the workshop, you will have several ROS bag files (`.bag`). To efficiently process and analyze these data in Python, it's recommended to convert the ROS bags into HDF5 (`.h5`) format.

For example, to convert a drone flight data bag file named `drone_hdr.bag` to an HDF5 file, you can run the following command:

```bash
python bag_to_h5.py drone_hdr.bag hdr_drone.h5 \
    --events_topic /capture_node/events \
    --image_topic /capture_node/camera/image \
    --start_offset 53.0 \
    --duration 5
```

It is recommended that you check the start_offset value by previewing the bag file in rqt_image_view or another ROS visualization tool. This helps you identify when the relevant data begins so you can skip any irrelevant initial parts of the recording.
Explanation of the command options:

- `drone_hdr.bag`: Path to the input ROS bag file.

- `hdr_drone.h5`: Path to the output HDF5 file.

- `--events_topic`: The ROS topic containing event data (default: /capture_node/events).

- `--image_topic`: The ROS topic containing image data (optional, in this example /capture_node/camera/image).

- `--start_offset`: Start processing data from this many seconds into the bag.

- `--duration`: Duration in seconds of data to process after the offset.

After running this command, you will have an HDF5 file ready for further analysis in Python.