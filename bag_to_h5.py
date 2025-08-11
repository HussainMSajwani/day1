#!/usr/bin/env python3
import os
import rosbag
import rospy
import numpy as np
import h5py
from typing import Optional, Tuple
from tqdm import tqdm

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

def arr_to_struct(arr):
    """Convert Nx4 float/int array into structured array with fields x, y, p, t"""
    out = np.zeros(len(arr), dtype=[('x', '<i2'), ('y', '<i2'), ('p', '?'), ('t', '<f8')])
    arr[:, 3] = arr[:, 3] - arr[0, 3]  # normalize time to start at zero
    out['x'] = arr[:, 0].astype(np.int16)
    out['y'] = arr[:, 1].astype(np.int16)
    out['p'] = arr[:, 2].astype(bool)
    out['t'] = arr[:, 3].astype(np.float64)
    return out

def convert_rosbag_to_h5(
    bag_path: str,
    output_h5_path: str,
    start_offset_s: float = 0.0,
    duration_s: Optional[float] = None,
    events_topic: str = "/capture_node/events",
    image_topic: Optional[str] = None,
) -> None:
    if not os.path.isfile(bag_path):
        raise FileNotFoundError(f"Bag not found: {bag_path}")

    with rosbag.Bag(bag_path, "r") as bag:
        start_time, end_time = _window_from_bag(bag, start_offset_s, duration_s)
        bag_start_time = start_time.to_sec()

        # Collect events
        events = []
        total_msgs = bag.get_message_count(topic_filters=[events_topic])

        for _, msg, _ in tqdm(bag.read_messages(topics=[events_topic], start_time=start_time, end_time=end_time),
                              total=total_msgs, desc="Reading events"):
            for ev in msg.events:
                t_sec = (float(ev.ts.secs) + float(ev.ts.nsecs) * 1e-9)
                events.append([int(ev.x), int(ev.y), int(ev.polarity), t_sec])

        if not events:
            print("No events found, writing empty dataset.")
            events_array = np.zeros((0,), dtype=[('x', '<i2'), ('y', '<i2'), ('p', '?'), ('t', '<f8')])
        else:
            events_array = arr_to_struct(np.array(events))

        with h5py.File(output_h5_path, "w") as h5f:
            h5f.create_dataset("events_data", data=events_array, compression="gzip", compression_opts=9)

            if image_topic:
                vlen_bytes = h5py.vlen_dtype(np.dtype("uint8"))
                vlen_str = h5py.string_dtype(encoding="utf-8")

                ts_ds = h5f.create_dataset("image_timestamps_s", shape=(0,), maxshape=(None,), dtype=np.float64)
                h_ds  = h5f.create_dataset("image_height",       shape=(0,), maxshape=(None,), dtype=np.int32)
                w_ds  = h5f.create_dataset("image_width",        shape=(0,), maxshape=(None,), dtype=np.int32)
                st_ds = h5f.create_dataset("image_step",         shape=(0,), maxshape=(None,), dtype=np.int32)
                enc_ds= h5f.create_dataset("image_encoding",     shape=(0,), maxshape=(None,), dtype=vlen_str)
                dat_ds= h5f.create_dataset("image_data",         shape=(0,), maxshape=(None,), dtype=vlen_bytes)

                count = 0
                img_total = bag.get_message_count(topic_filters=[image_topic])
                for _, msg, _ in tqdm(bag.read_messages(topics=[image_topic], start_time=start_time, end_time=end_time),
                                      total=img_total, desc="Reading images"):
                    count += 1
                    ts = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs) * 1e-9

                    ts_ds.resize((count,))
                    h_ds.resize((count,))
                    w_ds.resize((count,))
                    st_ds.resize((count,))
                    enc_ds.resize((count,))
                    dat_ds.resize((count,))

                    ts_ds[count - 1]  = ts - bag_start_time
                    h_ds[count - 1]   = int(msg.height)
                    w_ds[count - 1]   = int(msg.width)
                    st_ds[count - 1]  = int(msg.step)
                    enc_ds[count - 1] = getattr(msg, "encoding", "")
                    dat_ds[count - 1] = np.frombuffer(msg.data, dtype=np.uint8)

    print(f"✅ Success — wrote {output_h5_path} with {len(events_array)} events")

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
