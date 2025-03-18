# Import modules
from picamera2 import Picamera2
import cv2
import threading
import os
import time
import datetime
import numpy as np  # We need this for comparing images

# Initialize global camera variables
picam2 = None
output_frame = None

def init_video_writer(frame, fps, video_filename):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
                video_filename, fourcc, fps,(frame.shape[1], frame.shape[0]))
        return video_writer

def capture_record():
        # read in global variable
        global picam2
        
        # ===== STEP 1: Start the camera =====
        # This turns on the camera and gets it ready to take pictures
        picam2 = Picamera2()
        
        # set framerate
        framerate = 30
        
        # configure picamera2
        config = picam2.create_video_configuration(main={"size": (1920, 1080)})
        config["controls"] = {
                "FrameRate": framerate
                }
        picam2.configure(config)
        picam2.start()
        
        # Recording settings
        recording_duration = 60
        frames_to_record = (recording_duration * framerate)
        
        # Create time stamp for video file
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%H-%M-%S_%b-%d-%Y")
        date_today = current_time.strftime("%Y-%m-%d")
        
        # ===== STEP 2: Set up where to save our videos =====
        # We're telling the computer where to put the videos we record
        folder_path = f"/mnt/usb/{date_today}"
        
        # Create video folder
        if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Created folder: {folder_path}")
        
        video_filename = f"{folder_path}/{timestamp}.mp4"
        
        # ===== STEP 3: Motion Detection Setup =====
        # We need to take a first picture to compare with later pictures
        # Like taking a picture of a room when it's empty
        print("Getting ready to detect motion...")
        first_frame = picam2.capture_array()
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)  # Make it black and white
        first_frame = cv2.GaussianBlur(first_frame, (21, 21), 0)  # Make it a bit blurry
        
        # Wait a moment for the camera to settle
        time.sleep(2)
        print("Camera ready! Watching for motion...")
        
        # Get a frame to set up the video recording
        color_frame = picam2.capture_array()
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
        
        # Video recording variables
        video_writer = None
        is_recording = False
        frames_recorded = 0
        
        # ===== STEP 4: Start looking for motion =====
        # This is like watching a room to see if anything moves
        while True:
                # Take a new picture
                frame = picam2.capture_array()
                color_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Color version for recording
                
                # Make it black and white and blurry (easier to compare)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
                
                # ===== STEP 5: Compare the pictures =====
                # Find the difference between the first picture and this new one
                # Like playing "spot the difference" between two pictures
                frame_delta = cv2.absdiff(first_frame, gray_frame)
                
                # Make the differences more obvious - if pixels changed a lot, make them white
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                
                # Clean up the image to see changes better
                thresh = cv2.dilate(thresh, None, iterations=2)
                
                # Find the outlines of things that changed
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # ===== STEP 6: Check if something moved =====
                motion_detected = False
                
                # Look at each outline we found
                for contour in contours:
                        # Only care about big enough changes (ignore tiny movements)
                        if cv2.contourArea(contour) < 500:  # This number controls sensitivity
                                continue
                        
                        # If we get here, something big enough moved!
                        motion_detected = True
                        
                        # Draw a rectangle around what moved (so we can see it)
                        (x, y, w, h) = cv2.boundingRect(contour)
                        cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # ===== STEP 7: Start recording if motion was found =====
                # If we found motion and we're not already recording, start recording
                if motion_detected and not is_recording:
                        print("Motion detected! Starting to record...")
                        video_writer = init_video_writer(color_frame, framerate, video_filename)
                        is_recording = True
                        frames_recorded = 0
                
                # ===== STEP 8: Record video if we're in recording mode =====
                if is_recording:
                        # Write frame to video file
                        video_writer.write(color_frame)
                        frames_recorded += 1
                        
                        # Stop recording after we've recorded enough frames
                        if frames_recorded >= frames_to_record:
                                print(f"Recording stopped: {video_filename}")
                                video_writer.release()
                                
                                # Create a new video filename for next time
                                current_time = datetime.datetime.now()
                                timestamp = current_time.strftime("%H-%M-%S_%b-%d-%Y")
                                video_filename = f"{folder_path}/{timestamp}.mp4"
                                
                                # Reset recording state
                                is_recording = False
                                
                                # Update our reference frame after recording
                                first_frame = gray_frame.copy()
                
                # Add a delay to control the framerate
                time.sleep(1/framerate)
        
        # Cleanup
        if video_writer is not None:
                video_writer.release()
        picam2.stop()
        picam2.close()

if __name__ == '__main__':
        capture_record()
