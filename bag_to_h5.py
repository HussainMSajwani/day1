import rosbag
import numpy as np
import h5py
from tqdm import tqdm

bag_path = '/home/aric/adasi/data/2025-08-10-18-39-47.bag'
h5_path = '/home/aric/adasi/data/2025-08-10-18-39-47.h5'

chunk_size = 100000  # adjust based on available memory
events_buffer = []

# Detect topic
with rosbag.Bag(bag_path, 'r') as bag:
    topics = bag.get_type_and_topic_info()[1].keys()
    if '/capture_node/events' in topics:
        event_topic = '/capture_node/events'
    elif '/dvs/events' in topics:
        event_topic = '/dvs/events'
    else:
        raise ValueError("No events topic found in bag file.")
print(f"Using topic: {event_topic}")

# First pass: count messages for progress bar
print("Counting messages...")
with rosbag.Bag(bag_path, 'r') as bag:
    total_msgs = bag.get_message_count(topic_filters=[event_topic])

# Second pass: convert to HDF5 with tqdm
with h5py.File(h5_path, 'w') as h5f:
    # Add metadata
    h5f.attrs['source_bag'] = bag_path
    h5f.attrs['topic'] = event_topic
    h5f.attrs['chunk_size'] = chunk_size

    dset = h5f.create_dataset(
        'events_data',
        shape=(0, 4),
        maxshape=(None, 4),
        dtype=np.float64,
        chunks=(chunk_size, 4)
    )

    idx = 0
    with rosbag.Bag(bag_path, 'r') as bag:
        for _, msg, _ in tqdm(bag.read_messages(topics=[event_topic]),
                              total=total_msgs, unit="msg", desc="Converting"):
            for ev in msg.events:
                t_sec = ev.ts.secs + (ev.ts.nsecs / 1e9)
                events_buffer.append([ev.x, ev.y, ev.polarity, t_sec])

            if len(events_buffer) >= chunk_size:
                arr = np.array(events_buffer, dtype=np.float64)
                dset.resize((idx + arr.shape[0], 4))
                dset[idx:idx + arr.shape[0], :] = arr
                idx += arr.shape[0]
                events_buffer.clear()
                h5f.flush()  # Ensure data is saved

        # Write any leftovers
        if events_buffer:
            arr = np.array(events_buffer, dtype=np.float64)
            dset.resize((idx + arr.shape[0], 4))
            dset[idx:idx + arr.shape[0], :] = arr

print(f"✅ Done: Events saved to {h5_path}")
