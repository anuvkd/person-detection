import cv2
import numpy as np
import os
import tensorflow as tf
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

# Path to your video file
video_file = './person.mp4'

# Output directory for frames
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# GStreamer pipeline 
gst_pipeline = (
    f"filesrc location={video_file} ! "
    "decodebin ! "
    "videoconvert ! "
    "videoscale ! "
    "video/x-raw,format=RGB ! "
    "appsink name=appsink emit-signals=true sync=false"
)

# Load Faster R-CNN model
def load_model():
    model = tf.saved_model.load('./saved_model')
    return model

model = load_model()

# Create the GStreamer pipeline
pipeline = Gst.parse_launch(gst_pipeline)

# Get the appsink element
appsink = pipeline.get_by_name("appsink")
appsink.set_property('emit-signals', True)
appsink.set_property('sync', False)

frame_counter = 0

# Function to handle new samples from appsink
def new_sample(sink):
    global frame_counter
    
    sample = sink.emit('pull-sample')
    if not sample:
        print("Failed to get a sample.")
        return Gst.FlowReturn.ERROR

    buf = sample.get_buffer()
    caps = sample.get_caps()
    if not caps:
        print("Failed to get caps from sample")
        return Gst.FlowReturn.ERROR

    width = caps.get_structure(0).get_value('width')
    height = caps.get_structure(0).get_value('height')

    (ret, map_info) = buf.map(Gst.MapFlags.READ)
    if not ret:
        print("Failed to map buffer")
        return Gst.FlowReturn.ERROR

    frame_data = np.frombuffer(map_info.data, np.uint8)
    buf.unmap(map_info)

    frame = frame_data.reshape((height, width, 3))

    resized_frame = cv2.resize(frame, (640, 640))
    
    # Prepare the frame for inference
    input_tensor = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
    
    # Perform inference
    detections = model(input_tensor)
    
    # Process detections 
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()
    num_detections = int(detections['num_detections'][0])

    # Draw bounding boxes for detected persons
    for i in range(num_detections):
        if scores[i] > 0.5 and classes[i] == 1:  
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * 640)
            xmax = int(xmax * 640)
            ymin = int(ymin * 640)
            ymax = int(ymax * 640)
            
            cv2.rectangle(resized_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(resized_frame, f'Person: {scores[i]:.2f}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the frame to the output directory
    frame_filename = os.path.join(output_dir, f'frame_{frame_counter:05d}.png')
    cv2.imwrite(frame_filename, resized_frame)
    print(f"Saved frame {frame_counter} to {frame_filename}")
    
    frame_counter += 1

    return Gst.FlowReturn.OK

# Connect the callback function to the appsink
appsink.connect('new-sample', new_sample)

# Start the pipeline
print("Starting pipeline...")
pipeline.set_state(Gst.State.PLAYING)

# Run the GLib main loop

loop = GLib.MainLoop()
loop.run()

# Clean up
print("Stopping pipeline...")
pipeline.set_state(Gst.State.NULL)
cv2.destroyAllWindows()
print("Pipeline stopped and resources cleaned up.")
