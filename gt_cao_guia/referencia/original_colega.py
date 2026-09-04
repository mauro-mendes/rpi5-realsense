#!/usr/bin/env python3
"""
realsense_aruco_standalone.py - Test version without ROS dependencies
"""
import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import csv
from datetime import datetime
import time
import os

# Configuration
MODEL_NAME = "yolov8n.pt"
CONF_THRESH = 0.35
# Save inside utils_package/trajetorias (sibling of scripts)
CSV_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "trajetorias"))
ARUCO_MARKER_LENGTH = 0.15
GRAVITY_SAMPLES = 300
ASSOC_DISTANCE_THRESHOLD = 1.5  # meters - max distance to associate detection with existing track
MAX_MISSES = 10  # frames - remove track after this many misses
DETECT_EVERY_N_FRAMES = 1

class Track:
    """Simple tracking object for person trajectories."""
    next_id = 0
    
    def __init__(self, position, frame_num):
        self.id = Track.next_id
        Track.next_id += 1
        self.position = position  # [x, y, z] in world frame
        self.last_update = frame_num
        self.miss_count = 0
        self.hit_count = 1
    
    def update(self, position, frame_num):
        self.position = position
        self.last_update = frame_num
        self.miss_count = 0
        self.hit_count += 1
    
    def increment_miss(self):
        self.miss_count += 1
    
    def is_lost(self):
        return self.miss_count >= MAX_MISSES

class RealsenseArucoStandalone:
    def __init__(self):
        print("[RealsenseAruco] Initializing standalone mode...")
        
        # Create output directory
        os.makedirs(CSV_DIR, exist_ok=True)
        timestamp = int(time.time())
        self.csv_path = os.path.join(CSV_DIR, f"trajetoria_pessoa_{timestamp}.csv")
        
        print(f"[RealsenseAruco] Recording trajectory to {self.csv_path}")
        
        # Initialize YOLO (with offline support)
        # Use local model file from scripts directory for offline operation
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "yolov8n.pt")
        
        print(f"[RealsenseAruco] Loading YOLO: {model_path}")
        self.model = None
        try:
            self.model = YOLO(model_path)
            print("[RealsenseAruco] YOLO model loaded successfully")
        except ConnectionError as e:
            print(f"[RealsenseAruco] YOLO download failed (offline network): {e}")
            print("[RealsenseAruco] Continuing without YOLO detection. Please pre-download the model.")
            self.yolo_enabled = False
        except Exception as e:
            print(f"[RealsenseAruco] Failed to load YOLO: {e}")
            self.yolo_enabled = False
        
        if self.model is not None:
            self.yolo_enabled = True
        
        # Initialize ArUco (compatible with older OpenCV versions)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # Initialize RealSense
        print("[RealsenseAruco] Initializing RealSense camera...")
        self.pipeline = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.cfg.enable_stream(rs.stream.accel)
        
        self.profile = self.pipeline.start(self.cfg)
        self.align = rs.align(rs.stream.color)
        
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print(f"[RealsenseAruco] Depth scale: {self.depth_scale}")
        
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
        # Camera matrix for ArUco pose estimation
        self.camera_matrix = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.array(self.intrinsics.coeffs, dtype=np.float32)
        
        # CSV setup
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp_iso", "track_id", "x_world", "y_world", "z_world"])
        
        self.frame_count = 0
        self.aruco_transform = None  # Store ArUco marker transform
        self.aruco_detected = False  # Flag to track if ArUco has been found
        self.tracks = []  # Active tracks list
        self.pending_rows = []  # Buffered (timestamp_iso, track_id, x, y, z) rows in camera frame, awaiting first ArUco detection
        print("[RealsenseAruco] Initialization complete!")
    
    def run(self):
        print("[RealsenseAruco] Starting main loop. Press 'q' to quit.")
        try:
            while True:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Convert to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                self.frame_count += 1
                
                # Detect ArUco markers
                if self.frame_count % DETECT_EVERY_N_FRAMES == 0:
                    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                    corners, ids, rejected = cv2.aruco.detectMarkers(
                        gray, self.aruco_dict, parameters=self.aruco_params
                    )
                    
                    if ids is not None and len(ids) > 0:
                        # Draw detected markers
                        cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
                        
                        # Estimate pose of first marker
                        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                            corners, ARUCO_MARKER_LENGTH, self.camera_matrix, self.dist_coeffs
                        )
                        
                        # Use first detected marker as reference
                        rvec = rvecs[0][0]
                        tvec = tvecs[0][0]
                        
                        # Convert rotation vector to rotation matrix
                        R_marker, _ = cv2.Rodrigues(rvec)
                        
                        # Create transformation matrix (camera to marker)
                        T_cam_to_marker = np.eye(4)
                        T_cam_to_marker[:3, :3] = R_marker
                        T_cam_to_marker[:3, 3] = tvec
                        
                        # Invert to get marker to camera transform
                        self.aruco_transform = np.linalg.inv(T_cam_to_marker)
                        
                        # If this is the first detection, transform all existing tracks
                        if not self.aruco_detected:
                            self.aruco_detected = True
                            print(f"[ArUco] First detection at frame {self.frame_count}! Transforming {len(self.tracks)} existing tracks...")
                            for track in self.tracks:
                                # Transform track position from camera frame to marker frame
                                pos_cam_homo = np.array([track.position[0], track.position[1],
                                                        track.position[2], 1.0])
                                pos_world_homo = self.aruco_transform @ pos_cam_homo
                                track.position = pos_world_homo[:3].tolist()
                            print(f"[ArUco] Retroactive transformation complete")

                            # Retroactively correct CSV rows recorded before the marker was ever seen
                            print(f"[ArUco] Correcting {len(self.pending_rows)} buffered CSV rows recorded before detection...")
                            for timestamp_iso, track_id, x, y, z in self.pending_rows:
                                pos_cam_homo = np.array([x, y, z, 1.0])
                                pos_world_homo = self.aruco_transform @ pos_cam_homo
                                self.csv_writer.writerow([timestamp_iso, track_id, *pos_world_homo[:3].tolist()])
                            self.pending_rows = []
                            print(f"[ArUco] Buffered rows corrected and written")
                        
                        # Draw axis on marker
                        cv2.drawFrameAxes(color_image, self.camera_matrix, self.dist_coeffs, 
                                        rvec, tvec, ARUCO_MARKER_LENGTH * 0.5)
                        
                        print(f"Frame {self.frame_count}: ArUco marker at ({tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f})m")
                    
                # Run YOLO detection if available
                if self.yolo_enabled and self.model is not None:
                    results = self.model(color_image, conf=CONF_THRESH, verbose=False)
                    
                    # Collect all detections for this frame
                    detections = []
                    detection_boxes = []
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            if int(box.cls[0]) == 0:  # Person class
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])
                                
                                # Get center point
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                
                                # Get 3D coordinates
                                depth_value = depth_frame.get_distance(cx, cy)
                                
                                if depth_value > 0:
                                    # Deproject to 3D camera coordinates
                                    point_3d_cam = rs.rs2_deproject_pixel_to_point(
                                        self.intrinsics, [cx, cy], depth_value
                                    )
                                    
                                    # Transform to ArUco marker frame if available
                                    if self.aruco_transform is not None:
                                        # Convert to homogeneous coordinates
                                        point_cam_homo = np.array([point_3d_cam[0], point_3d_cam[1], 
                                                                   point_3d_cam[2], 1.0])
                                        # Transform to marker frame
                                        point_world_homo = self.aruco_transform @ point_cam_homo
                                        point_world = point_world_homo[:3]
                                    else:
                                        # No marker detected yet, use camera frame
                                        point_world = point_3d_cam
                                    
                                    detections.append(point_world)
                                    detection_boxes.append((x1, y1, x2, y2, conf))
                    
                    # Track association
                    matched_tracks = set()
                    matched_detections = set()
                    
                    # For each detection, find closest track
                    for det_idx, detection in enumerate(detections):
                        best_track_idx = -1
                        best_distance = ASSOC_DISTANCE_THRESHOLD
                        
                        for track_idx, track in enumerate(self.tracks):
                            if track_idx in matched_tracks:
                                continue
                            
                            # Calculate 3D distance
                            dist = np.linalg.norm(np.array(detection) - np.array(track.position))
                            
                            if dist < best_distance:
                                best_distance = dist
                                best_track_idx = track_idx
                        
                        if best_track_idx >= 0:
                            # Update existing track
                            self.tracks[best_track_idx].update(detection, self.frame_count)
                            matched_tracks.add(best_track_idx)
                            matched_detections.add(det_idx)
                        else:
                            # Create new track
                            new_track = Track(detection, self.frame_count)
                            self.tracks.append(new_track)
                            matched_tracks.add(len(self.tracks) - 1)
                            matched_detections.add(det_idx)
                            print(f"[Track] New track {new_track.id} created at ({detection[0]:.2f}, {detection[1]:.2f}, {detection[2]:.2f})")
                    
                    # Increment miss count for unmatched tracks
                    for track_idx, track in enumerate(self.tracks):
                        if track_idx not in matched_tracks:
                            track.increment_miss()
                    
                    # Remove lost tracks
                    self.tracks = [t for t in self.tracks if not t.is_lost()]
                    
                    # Log all active tracks to CSV and draw
                    timestamp_iso = datetime.now().isoformat()
                    for track_idx, track in enumerate(self.tracks):
                        # Save to CSV (buffer until the marker frame is established, so
                        # early camera-frame rows can be retroactively corrected)
                        if self.aruco_detected:
                            self.csv_writer.writerow([
                                timestamp_iso, track.id,
                                track.position[0], track.position[1], track.position[2]
                            ])
                        else:
                            self.pending_rows.append((
                                timestamp_iso, track.id,
                                track.position[0], track.position[1], track.position[2]
                            ))
                        
                        # Draw on image (find corresponding detection box if matched)
                        if track_idx in matched_tracks:
                            det_idx = list(matched_detections)[list(matched_tracks).index(track_idx)]
                            x1, y1, x2, y2, conf = detection_boxes[det_idx]
                            
                            # Color based on track ID
                            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                            color = colors[track.id % len(colors)]
                            
                            cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                            label = f"ID:{track.id} ({conf:.2f})"
                            if self.aruco_transform is not None:
                                label += f" @({track.position[0]:.2f},{track.position[1]:.2f})m"
                            cv2.putText(color_image, label, 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, color, 2)
                elif not self.yolo_enabled:
                    if self.frame_count % 300 == 0:
                        print("[RealsenseAruco] YOLO disabled - no person detection available")
                
                # Display
                cv2.imshow('RealSense ArUco + YOLO', color_image)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[RealsenseAruco] Quit requested")
                    break
                    
        except KeyboardInterrupt:
            print("[RealsenseAruco] Interrupted by user")
        finally:
            self.shutdown()
    
    def shutdown(self):
        print("[RealsenseAruco] Shutting down and saving CSV...")
        if self.pending_rows:
            print(f"[RealsenseAruco] ArUco marker was never detected; writing {len(self.pending_rows)} rows in raw camera frame")
            for row in self.pending_rows:
                self.csv_writer.writerow(row)
            self.pending_rows = []
        self.csv_file.close()
        self.pipeline.stop()
        cv2.destroyAllWindows()
        print(f"[RealsenseAruco] Data saved to {self.csv_path}")

if __name__ == "__main__":
    node = RealsenseArucoStandalone()
    node.run()
