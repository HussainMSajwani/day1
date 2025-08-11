#!/usr/bin/env python3
import os
import rosbag
import rospy
import numpy as np
import h5py
from typing import Optional, Tuple

def _to_ros_time(sec: float) -> rospy.Time:
    return rospy.Time.from_sec(float(sec))

def _window_from_bag(bag: rosbag.Bag,
                     start_offset: float,
                     duration: Optional[float]) -> Tuple[rospy.Time, Optional[rospy.Time]]:
    bag_start = bag.get_start_time()  # epoch seconds (float)
    start_time = _to_ros_time(bag_start + float(start_offset))
    end_time = None
    if duration is not None:
        end_time = _to_ros_time(bag_start + float(start_offset) + float(duration))
    return start_time, end_time

def convert_rosbag_to_h5(
    bag_path: str,
    output_h5_path: str,
    start_offset_s: float = 0.0,
    duration_s: Optional[float] = None,
    events_topic: str = "/capture_node/events",
    image_topic: Optional[str] = None,
) -> None:
    """
    Convert a ROS bag to HDF5 with the SAME 'events_data' dataset structure as the old script.

    - events_data: Nx4 array [x, y, polarity, t_sec]  (exactly like the old code)
    - If image_topic is provided, the following extra datasets are created:
        image_timestamps_s : (M,) float64
        image_height       : (M,) int32
        image_width        : (M,) int32
        image_step         : (M,) int32
        image_encoding     : (M,) UTF-8 variable length strings
        image_data         : (M,) variable length uint8 blobs (raw ROS bytes)
    """
    if not os.path.isfile(bag_path):
        raise FileNotFoundError(f"Bag not found: {bag_path}")

    with rosbag.Bag(bag_path, "r") as bag:
        start_time, end_time = _window_from_bag(bag, start_offset_s, duration_s)

        # ---- Collect events (exact structure)
        events = []
        for topic, msg, t in bag.read_messages(topics=[events_topic], start_time=start_time, end_time=end_time):
            # Expecting msg.events with fields: x, y, polarity, ts(secs,nsecs)
            for ev in msg.events:
                t_sec = float(ev.ts.secs) + float(ev.ts.nsecs) * 1e-9
                events.append([int(ev.x), int(ev.y), int(ev.polarity), t_sec])

        events_array = np.array(events, dtype=np.float64) if events else np.zeros((0, 4), dtype=np.float64)

        # ---- Write H5 (keep 'events_data' exactly as before)
        with h5py.File(output_h5_path, "w") as h5f:
            # EXACT dataset name + layout
            h5f.create_dataset("events_data", data=events_array)

            # ---- Optional: images (added as new datasets; does not affect events_data)
            if image_topic:
                # variable-length dtypes
                vlen_bytes = h5py.vlen_dtype(np.dtype("uint8"))
                vlen_str = h5py.string_dtype(encoding="utf-8")

                # create empty, resizable datasets
                ts_ds = h5f.create_dataset("image_timestamps_s", shape=(0,), maxshape=(None,), dtype=np.float64)
                h_ds  = h5f.create_dataset("image_height",       shape=(0,), maxshape=(None,), dtype=np.int32)
                w_ds  = h5f.create_dataset("image_width",        shape=(0,), maxshape=(None,), dtype=np.int32)
                st_ds = h5f.create_dataset("image_step",         shape=(0,), maxshape=(None,), dtype=np.int32)
                enc_ds= h5f.create_dataset("image_encoding",     shape=(0,), maxshape=(None,), dtype=vlen_str)
                dat_ds= h5f.create_dataset("image_data",         shape=(0,), maxshape=(None,), dtype=vlen_bytes)

                count = 0
                for topic, msg, t in bag.read_messages(topics=[image_topic], start_time=start_time, end_time=end_time):
                    count += 1
                    ts = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs) * 1e-9

                    # grow each dataset by 1 and append
                    ts_ds.resize((count,))
                    h_ds.resize((count,))
                    w_ds.resize((count,))
                    st_ds.resize((count,))
                    enc_ds.resize((count,))
                    dat_ds.resize((count,))

                    ts_ds[count - 1]  = ts
                    h_ds[count - 1]   = int(msg.height)
                    w_ds[count - 1]   = int(msg.width)
                    st_ds[count - 1]  = int(msg.step)
                    enc_ds[count - 1] = getattr(msg, "encoding", "")
                    dat_ds[count - 1] = np.frombuffer(msg.data, dtype=np.uint8)

    print(f"Success — wrote {output_h5_path} (events: {events_array.shape[0]})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert ROS bag to HDF5 with 'events_data' layout.")
    parser.add_argument("bag", help="Path to input .bag")
    parser.add_argument("out", help="Path to output .h5")
    parser.add_argument("--start_offset", type=float, default=0.0, help="Start offset from bag start (sec)")
    parser.add_argument("--duration", type=float, default=None, help="Duration to process (sec), default: until end")
    parser.add_argument("--events_topic", type=str, default="/capture_node/events", help="Events topic")
    parser.add_argument("--image_topic", type=str, default=None, help="Optional image topic to also log")
    args = parser.parse_args()

    convert_rosbag_to_h5(
        bag_path=args.bag,
        output_h5_path=args.out,
        start_offset_s=args.start_offset,
        duration_s=args.duration,
        events_topic=args.events_topic,
        image_topic=args.image_topic,
    )
