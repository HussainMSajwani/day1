#%%
# %matplotlib qt
import h5py
import numpy as np
import matplotlib.pyplot as plt


H, W = 480, 640

# drone_data = h5py.File('/home/aric/adasi/data/2025-07-17-12-49-53_full_testing.h5', 'r')
shapes_data = h5py.File('/home/aric/adasi/data/shapes.h5', 'r')

# drone_events = drone_data['events_data'][:]
events = shapes_data['events_data'][:]

print(events.dtype)
print("We have {} events".format(len(events)))

def get_events_between(events, t0, t1):
    idx = np.where((events['t'] >= t0) & (events['t'] <= t1))[0]
    out = events[idx]
    return out

t0 = 2.9
t1 = t0 + 0.03 #a 30 ms window

sample = get_events_between(events, t0, t1)

#3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sample['t'], sample['x'], sample['y'], c=sample['p'], s=1, cmap='coolwarm')
plt.title('Event cloud')
ax.set_xlabel('Time (us)')
ax.set_ylabel('X')
ax.set_zlabel('Y')

#%% Projection

img = np.zeros((H, W, 1), dtype=np.uint8) # W x H x C
# Initialize an empty image with 3 channels (RGB)
# Note: The dimensions (240, 346) should match the resolution of your event camera

for event in sample:
    x, y, p = event['x'], event['y'], event['p']

    img[y, x, 0] = 255  
    # if p:  # Positive event
    #     img[y, x, 0] = 255
    #     img[y, x, 1] = 0
    #     img[y, x, 2] = 0
    # else:  # Negative event
    #     img[y, x, 0] = 0
    #     img[y, x, 1] = 0
    #     img[y, x, 2] = 255

plt.figure(figsize=(15, 10))
plt.imshow(img, cmap='gray')
plt.xlabel('X (px)')
plt.ylabel('Y (px)')
plt.title('Events between {} and {} seconds'.format(t0, t1))
plt.show()

#%% histogram

img = np.zeros((H, W, 1), dtype=np.uint8)

for event in sample:
    x, y, p = event['x'], event['y'], event['p']
    img[y, x, 0] += 1

img = img / img.max() * 255  # Normalize to 0-255 range
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img, cmap='gray')
plt.xlabel('X (px)')
plt.ylabel('Y (px)')
plt.title('Histogram of events')
plt.show()

# %%
frame = np.zeros((H, W, 2))

for e in sample:
    frame[e['y'], e['x'], e['p']*1] += 1


frame = np.concatenate([np.zeros((H, W, 1)), frame], axis=2)
print(frame.shape)
fig, ax = plt.subplots(1, 1)

ax.imshow(frame, cmap='gray')
plt.title('Histogram of events by polarity')
# %% Bonus voxel grid
# %%
tau = 1e-3  # seconds (3 ms)

# Convert t_ref to seconds
t_ref = sample['t'][-1] 

# Initialize time surface (single channel)
time_surface = np.zeros((H, W), dtype=np.float32)

# Track last event time per pixel (in seconds)
last_time = np.full((H, W), -np.inf, dtype=np.float32)

# Fill in last_time from the event list
for e in sample:
    last_time[e['y'], e['x']] = e['t'] 

# Compute exponential decay for each pixel
valid_mask = last_time > -np.inf
time_surface[valid_mask] = np.exp(-(t_ref - last_time[valid_mask]) / tau)

# Display
fig, ax = plt.subplots(1, 1)
ax.imshow(time_surface, cmap='gray')
plt.title(r'Time Surface, $\tau = {:.1f}$ ms'.format(tau * 1e3))
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# --- Average Timestamp Surface ---

# If your timestamps are in microseconds and you want seconds, set convert_to_seconds=True
convert_to_seconds = False

sum_time   = np.zeros((H, W), dtype=np.float64)
count_time = np.zeros((H, W), dtype=np.int32)

for e in sample:
    t = e['t'] * 1e-6 if convert_to_seconds else e['t']
    sum_time[e['y'], e['x']]   += t
    count_time[e['y'], e['x']] += 1

avg_ts = np.zeros((H, W), dtype=np.float32)
valid  = count_time > 0
avg_ts[valid] = (sum_time[valid] / count_time[valid]).astype(np.float32)

avg_ts_norm = avg_ts / avg_ts.max() * 255

# Plot
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
im = ax.imshow(avg_ts_norm, cmap='gray')
ax.set_title('Average Timestamp Surface (normalized)')
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show()


# %%
plt.show()