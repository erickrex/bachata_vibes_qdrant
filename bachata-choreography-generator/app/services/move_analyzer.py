"""
Move analysis service using MediaPipe for pose detection and feature extraction.
Analyzes Bachata move clips to extract movement patterns and features.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PoseFeatures:
    """Container for pose analysis features from a single frame."""
    landmarks: np.ndarray  # 33 pose landmarks (x, y, z, visibility)
    joint_angles: Dict[str, float]  # Key joint angles
    center_of_mass: Tuple[float, float]  # Center of mass position
    bounding_box: Tuple[float, float, float, float]  # x, y, width, height
    confidence: float  # Overall pose detection confidence


@dataclass
class HandFeatures:
    """Container for hand analysis features from a single frame."""
    left_hand_landmarks: Optional[np.ndarray]  # 21 left hand landmarks
    right_hand_landmarks: Optional[np.ndarray]  # 21 right hand landmarks
    left_hand_confidence: float
    right_hand_confidence: float


@dataclass
class MovementDynamics:
    """Container for movement dynamics analysis across frames."""
    velocity_profile: np.ndarray  # Movement velocity over time
    acceleration_profile: np.ndarray  # Movement acceleration over time
    spatial_coverage: float  # Area covered by movement
    rhythm_score: float  # Rhythmic consistency score
    complexity_score: float  # Movement complexity score
    dominant_movement_direction: str  # Primary movement direction
    energy_level: str  # Estimated energy level (low/medium/high)
    
    # Enhanced movement dynamics
    footwork_area_coverage: float  # Area covered by foot movements
    upper_body_movement_range: float  # Range of upper body movement
    rhythm_compatibility_score: float  # Compatibility with musical rhythm
    movement_periodicity: float  # Regularity of movement patterns
    transition_points: List[int]  # Frame indices of movement transitions
    movement_intensity_profile: np.ndarray  # Intensity variation over time
    spatial_distribution: Dict[str, float]  # Movement distribution by body region


@dataclass
class MoveAnalysisResult:
    """Complete analysis result for a move clip."""
    video_path: str
    duration: float
    frame_count: int
    fps: float
    
    # Pose analysis
    pose_features: List[PoseFeatures]
    hand_features: List[HandFeatures]
    
    # Movement dynamics
    movement_dynamics: MovementDynamics
    
    # Feature embeddings
    pose_embedding: np.ndarray  # 384-dimensional pose feature vector
    movement_embedding: np.ndarray  # Movement pattern embedding
    
    # Enhanced analysis scores
    movement_complexity_score: float  # Movement complexity (0-1)
    tempo_compatibility_range: Tuple[float, float]  # (min_bpm, max_bpm)
    difficulty_score: float  # Difficulty level (0-1)
    
    # Quality metrics
    analysis_quality: float  # Overall analysis quality score
    pose_detection_rate: float  # Percentage of frames with successful pose detection


class MoveAnalyzer:
    """
    MediaPipe-based move analyzer for Bachata dance clips.
    Extracts pose landmarks, hand tracking, and movement dynamics.
    """
    
    def __init__(self, 
                 target_fps: int = 30,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the MoveAnalyzer.
        
        Args:
            target_fps: Target frame rate for analysis (frames sampled per second)
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.target_fps = target_fps
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe solutions
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize pose detector
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # Initialize hand detector
        self.hand_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        logger.info(f"MoveAnalyzer initialized with target_fps={target_fps}")
    
    def analyze_move_clip(self, video_path: str) -> MoveAnalysisResult:
        """
        Analyze a complete move clip and extract all features.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            MoveAnalysisResult containing all extracted features
        """
        logger.info(f"Analyzing move clip: {video_path}")
        
        # Load video and get basic info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps
        
        logger.info(f"Video info: {duration:.2f}s, {total_frames} frames, {original_fps:.1f} fps")
        
        # Calculate frame sampling
        frame_interval = max(1, int(original_fps / self.target_fps))
        sampled_frames = list(range(0, total_frames, frame_interval))
        
        logger.info(f"Sampling {len(sampled_frames)} frames at {frame_interval} frame intervals")
        
        # Extract features from sampled frames
        pose_features = []
        hand_features = []
        
        with tqdm(total=len(sampled_frames), desc="Analyzing frames") as pbar:
            for frame_idx in sampled_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Could not read frame {frame_idx}")
                    continue
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract pose features
                pose_feat = self._extract_pose_features(rgb_frame)
                if pose_feat:
                    pose_features.append(pose_feat)
                
                # Extract hand features
                hand_feat = self._extract_hand_features(rgb_frame)
                hand_features.append(hand_feat)
                
                pbar.update(1)
        
        cap.release()
        
        if not pose_features:
            raise ValueError(f"No pose features could be extracted from {video_path}")
        
        logger.info(f"Extracted features from {len(pose_features)} frames")
        
        # Calculate movement dynamics
        movement_dynamics = self._calculate_movement_dynamics(pose_features)
        
        # Generate embeddings
        pose_embedding = self._generate_pose_embedding(pose_features)
        movement_embedding = self._generate_movement_embedding(movement_dynamics)
        
        # Calculate enhanced analysis scores
        movement_complexity_score = self.calculate_movement_complexity_score(pose_features, movement_dynamics)
        tempo_compatibility_range = self.calculate_tempo_compatibility_range(movement_dynamics, pose_features)
        difficulty_score = self.calculate_difficulty_score(movement_dynamics, pose_features)
        
        # Calculate quality metrics
        pose_detection_rate = len(pose_features) / len(sampled_frames)
        analysis_quality = self._calculate_analysis_quality(pose_features, hand_features)
        
        result = MoveAnalysisResult(
            video_path=video_path,
            duration=duration,
            frame_count=len(sampled_frames),
            fps=self.target_fps,
            pose_features=pose_features,
            hand_features=hand_features,
            movement_dynamics=movement_dynamics,
            pose_embedding=pose_embedding,
            movement_embedding=movement_embedding,
            movement_complexity_score=movement_complexity_score,
            tempo_compatibility_range=tempo_compatibility_range,
            difficulty_score=difficulty_score,
            analysis_quality=analysis_quality,
            pose_detection_rate=pose_detection_rate
        )
        
        logger.info(f"Analysis complete. Quality: {analysis_quality:.2f}, "
                   f"Detection rate: {pose_detection_rate:.2f}")
        
        return result
    
    def _extract_pose_features(self, rgb_frame: np.ndarray) -> Optional[PoseFeatures]:
        """Extract pose features from a single frame."""
        results = self.pose_detector.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks
        landmarks = np.array([
            [lm.x, lm.y, lm.z, lm.visibility] 
            for lm in results.pose_landmarks.landmark
        ])
        
        # Calculate joint angles
        joint_angles = self._calculate_joint_angles(landmarks)
        
        # Calculate center of mass
        center_of_mass = self._calculate_center_of_mass(landmarks)
        
        # Calculate bounding box
        bounding_box = self._calculate_bounding_box(landmarks)
        
        # Calculate overall confidence
        confidence = np.mean(landmarks[:, 3])  # Average visibility scores
        
        return PoseFeatures(
            landmarks=landmarks,
            joint_angles=joint_angles,
            center_of_mass=center_of_mass,
            bounding_box=bounding_box,
            confidence=confidence
        )
    
    def _extract_hand_features(self, rgb_frame: np.ndarray) -> HandFeatures:
        """Extract hand features from a single frame."""
        results = self.hand_detector.process(rgb_frame)
        
        left_hand_landmarks = None
        right_hand_landmarks = None
        left_confidence = 0.0
        right_confidence = 0.0
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Extract landmarks
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                ])
                
                # Determine if left or right hand
                hand_label = handedness.classification[0].label
                confidence = handedness.classification[0].score
                
                if hand_label == "Left":
                    left_hand_landmarks = landmarks
                    left_confidence = confidence
                else:
                    right_hand_landmarks = landmarks
                    right_confidence = confidence
        
        return HandFeatures(
            left_hand_landmarks=left_hand_landmarks,
            right_hand_landmarks=right_hand_landmarks,
            left_hand_confidence=left_confidence,
            right_hand_confidence=right_confidence
        )
    
    def _calculate_joint_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate key joint angles from pose landmarks."""
        def angle_between_points(p1, p2, p3):
            """Calculate angle at p2 formed by p1-p2-p3."""
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.degrees(np.arccos(cos_angle))
        
        # MediaPipe pose landmark indices
        # Key joints for dance analysis
        joint_angles = {}
        
        try:
            # Left arm angles
            left_shoulder = landmarks[11, :3]  # LEFT_SHOULDER
            left_elbow = landmarks[13, :3]     # LEFT_ELBOW
            left_wrist = landmarks[15, :3]     # LEFT_WRIST
            
            joint_angles['left_elbow'] = angle_between_points(left_shoulder, left_elbow, left_wrist)
            
            # Right arm angles
            right_shoulder = landmarks[12, :3]  # RIGHT_SHOULDER
            right_elbow = landmarks[14, :3]     # RIGHT_ELBOW
            right_wrist = landmarks[16, :3]     # RIGHT_WRIST
            
            joint_angles['right_elbow'] = angle_between_points(right_shoulder, right_elbow, right_wrist)
            
            # Hip angles
            left_hip = landmarks[23, :3]   # LEFT_HIP
            left_knee = landmarks[25, :3]  # LEFT_KNEE
            left_ankle = landmarks[27, :3] # LEFT_ANKLE
            
            joint_angles['left_knee'] = angle_between_points(left_hip, left_knee, left_ankle)
            
            right_hip = landmarks[24, :3]   # RIGHT_HIP
            right_knee = landmarks[26, :3]  # RIGHT_KNEE
            right_ankle = landmarks[28, :3] # RIGHT_ANKLE
            
            joint_angles['right_knee'] = angle_between_points(right_hip, right_knee, right_ankle)
            
            # Torso angle (spine alignment)
            nose = landmarks[0, :3]
            left_shoulder = landmarks[11, :3]
            right_shoulder = landmarks[12, :3]
            mid_hip = (landmarks[23, :3] + landmarks[24, :3]) / 2
            
            # Calculate torso lean
            torso_vector = nose - mid_hip
            vertical_vector = np.array([0, -1, 0])
            joint_angles['torso_lean'] = angle_between_points(
                nose + vertical_vector, nose, mid_hip
            )
            
        except (IndexError, ValueError) as e:
            logger.warning(f"Error calculating joint angles: {e}")
            # Return default angles if calculation fails
            joint_angles = {
                'left_elbow': 180.0,
                'right_elbow': 180.0,
                'left_knee': 180.0,
                'right_knee': 180.0,
                'torso_lean': 0.0
            }
        
        return joint_angles
    
    def _calculate_center_of_mass(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """Calculate center of mass from pose landmarks."""
        # Use key body points for center of mass calculation
        key_points = [
            11, 12,  # Shoulders
            23, 24,  # Hips
            25, 26,  # Knees
        ]
        
        valid_points = landmarks[key_points, :2]  # x, y coordinates
        center_x = np.mean(valid_points[:, 0])
        center_y = np.mean(valid_points[:, 1])
        
        return (center_x, center_y)
    
    def _calculate_bounding_box(self, landmarks: np.ndarray) -> Tuple[float, float, float, float]:
        """Calculate bounding box around the pose."""
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        
        width = max_x - min_x
        height = max_y - min_y
        
        return (min_x, min_y, width, height)
    
    def _calculate_movement_dynamics(self, pose_features: List[PoseFeatures]) -> MovementDynamics:
        """Calculate enhanced movement dynamics from pose sequence."""
        if len(pose_features) < 2:
            # Return default dynamics for single frame
            return MovementDynamics(
                velocity_profile=np.array([0.0]),
                acceleration_profile=np.array([0.0]),
                spatial_coverage=0.0,
                rhythm_score=0.0,
                complexity_score=0.0,
                dominant_movement_direction="static",
                energy_level="low",
                footwork_area_coverage=0.0,
                upper_body_movement_range=0.0,
                rhythm_compatibility_score=0.0,
                movement_periodicity=0.0,
                transition_points=[],
                movement_intensity_profile=np.array([0.0]),
                spatial_distribution={"upper_body": 0.0, "lower_body": 0.0, "arms": 0.0, "legs": 0.0}
            )
        
        # Extract center of mass trajectory
        centers = np.array([pf.center_of_mass for pf in pose_features])
        
        # Calculate velocity and acceleration profiles
        velocities = np.diff(centers, axis=0)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        accelerations = np.diff(velocities, axis=0)
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        # Calculate spatial coverage
        x_range = np.max(centers[:, 0]) - np.min(centers[:, 0])
        y_range = np.max(centers[:, 1]) - np.min(centers[:, 1])
        spatial_coverage = x_range * y_range
        
        # Calculate rhythm score (consistency of movement)
        velocity_std = np.std(velocity_magnitudes)
        velocity_mean = np.mean(velocity_magnitudes)
        rhythm_score = 1.0 / (1.0 + velocity_std / (velocity_mean + 1e-6))
        
        # Calculate complexity score
        joint_angle_variations = []
        for angle_name in ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']:
            angles = [pf.joint_angles.get(angle_name, 180.0) for pf in pose_features]
            joint_angle_variations.append(np.std(angles))
        
        complexity_score = np.mean(joint_angle_variations) / 180.0  # Normalize to 0-1
        
        # Determine dominant movement direction
        total_displacement = centers[-1] - centers[0]
        if abs(total_displacement[0]) > abs(total_displacement[1]):
            dominant_direction = "horizontal" if total_displacement[0] > 0 else "horizontal_left"
        else:
            dominant_direction = "vertical_up" if total_displacement[1] < 0 else "vertical_down"
        
        # Estimate energy level
        avg_velocity = np.mean(velocity_magnitudes)
        if avg_velocity < 0.01:
            energy_level = "low"
        elif avg_velocity < 0.03:
            energy_level = "medium"
        else:
            energy_level = "high"
        
        # Enhanced movement dynamics analysis
        footwork_area = self._calculate_footwork_area_coverage(pose_features)
        upper_body_range = self._calculate_upper_body_movement_range(pose_features)
        rhythm_compatibility = self._calculate_rhythm_compatibility_score(pose_features)
        periodicity = self._calculate_movement_periodicity(velocity_magnitudes)
        transition_points = self._identify_transition_points(velocity_magnitudes, acceleration_magnitudes)
        intensity_profile = self._calculate_movement_intensity_profile(pose_features)
        spatial_dist = self._calculate_spatial_distribution(pose_features)
        
        return MovementDynamics(
            velocity_profile=velocity_magnitudes,
            acceleration_profile=acceleration_magnitudes,
            spatial_coverage=spatial_coverage,
            rhythm_score=rhythm_score,
            complexity_score=complexity_score,
            dominant_movement_direction=dominant_direction,
            energy_level=energy_level,
            footwork_area_coverage=footwork_area,
            upper_body_movement_range=upper_body_range,
            rhythm_compatibility_score=rhythm_compatibility,
            movement_periodicity=periodicity,
            transition_points=transition_points,
            movement_intensity_profile=intensity_profile,
            spatial_distribution=spatial_dist
        )
    
    def _calculate_footwork_area_coverage(self, pose_features: List[PoseFeatures]) -> float:
        """Calculate the area covered by foot movements."""
        if len(pose_features) < 2:
            return 0.0
        
        # Extract foot positions (ankles and feet)
        left_ankle_positions = []
        right_ankle_positions = []
        left_foot_positions = []
        right_foot_positions = []
        
        for pf in pose_features:
            landmarks = pf.landmarks
            # MediaPipe landmark indices: 27=LEFT_ANKLE, 28=RIGHT_ANKLE, 31=LEFT_FOOT_INDEX, 32=RIGHT_FOOT_INDEX
            if landmarks.shape[0] > 32:
                left_ankle_positions.append(landmarks[27, :2])  # x, y
                right_ankle_positions.append(landmarks[28, :2])
                left_foot_positions.append(landmarks[31, :2])
                right_foot_positions.append(landmarks[32, :2])
        
        if not left_ankle_positions:
            return 0.0
        
        # Combine all foot-related positions
        all_foot_positions = np.vstack([
            left_ankle_positions, right_ankle_positions,
            left_foot_positions, right_foot_positions
        ])
        
        # Calculate convex hull area
        try:
            from scipy.spatial import ConvexHull
            if len(all_foot_positions) >= 3:
                hull = ConvexHull(all_foot_positions)
                return hull.volume  # In 2D, volume is area
        except:
            # Fallback to bounding box area
            x_range = np.max(all_foot_positions[:, 0]) - np.min(all_foot_positions[:, 0])
            y_range = np.max(all_foot_positions[:, 1]) - np.min(all_foot_positions[:, 1])
            return x_range * y_range
        
        return 0.0
    
    def _calculate_upper_body_movement_range(self, pose_features: List[PoseFeatures]) -> float:
        """Calculate the range of upper body movement."""
        if len(pose_features) < 2:
            return 0.0
        
        # Extract upper body key points (shoulders, elbows, wrists)
        upper_body_positions = []
        
        for pf in pose_features:
            landmarks = pf.landmarks
            # Upper body landmarks: 11,12=shoulders, 13,14=elbows, 15,16=wrists
            upper_points = landmarks[[11, 12, 13, 14, 15, 16], :2]  # x, y coordinates
            upper_body_positions.append(upper_points.flatten())
        
        upper_body_positions = np.array(upper_body_positions)
        
        # Calculate movement range as standard deviation across time
        movement_ranges = np.std(upper_body_positions, axis=0)
        return np.mean(movement_ranges)
    
    def _calculate_rhythm_compatibility_score(self, pose_features: List[PoseFeatures]) -> float:
        """Calculate rhythm compatibility score by analyzing movement timing patterns."""
        if len(pose_features) < 4:
            return 0.0
        
        # Extract movement velocity from center of mass
        centers = np.array([pf.center_of_mass for pf in pose_features])
        velocities = np.diff(centers, axis=0)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        # Analyze periodicity using autocorrelation
        if len(velocity_magnitudes) < 4:
            return 0.0
        
        # Normalize velocity signal
        velocity_norm = (velocity_magnitudes - np.mean(velocity_magnitudes)) / (np.std(velocity_magnitudes) + 1e-6)
        
        # Calculate autocorrelation
        autocorr = np.correlate(velocity_norm, velocity_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation (indicating rhythm)
        if len(autocorr) > 1:
            # Look for secondary peaks (excluding the first peak at lag 0)
            if len(autocorr) > 3:
                secondary_peaks = autocorr[2:min(len(autocorr), len(velocity_magnitudes)//2)]
                if len(secondary_peaks) > 0:
                    max_secondary_peak = np.max(secondary_peaks)
                    return max(0.0, max_secondary_peak / autocorr[0])  # Normalize by first peak
        
        return 0.0
    
    def _calculate_movement_periodicity(self, velocity_magnitudes: np.ndarray) -> float:
        """Calculate the periodicity/regularity of movement patterns."""
        if len(velocity_magnitudes) < 4:
            return 0.0
        
        # Use FFT to find dominant frequencies
        try:
            fft = np.fft.fft(velocity_magnitudes - np.mean(velocity_magnitudes))
            power_spectrum = np.abs(fft[:len(fft)//2])
            
            if len(power_spectrum) > 1:
                # Find the ratio of the dominant frequency to total power
                max_power = np.max(power_spectrum[1:])  # Exclude DC component
                total_power = np.sum(power_spectrum[1:])
                return max_power / (total_power + 1e-6)
        except:
            pass
        
        return 0.0
    
    def _identify_transition_points(self, velocity_magnitudes: np.ndarray, 
                                  acceleration_magnitudes: np.ndarray) -> List[int]:
        """Identify transition points in movement based on velocity and acceleration changes."""
        if len(velocity_magnitudes) < 3:
            return []
        
        transition_points = []
        
        # Find points where acceleration changes significantly
        if len(acceleration_magnitudes) > 2:
            # Calculate acceleration change rate
            accel_changes = np.abs(np.diff(acceleration_magnitudes))
            
            # Find peaks in acceleration changes
            threshold = np.mean(accel_changes) + np.std(accel_changes)
            
            for i in range(1, len(accel_changes) - 1):
                if (accel_changes[i] > threshold and 
                    accel_changes[i] > accel_changes[i-1] and 
                    accel_changes[i] > accel_changes[i+1]):
                    transition_points.append(i + 1)  # Adjust for diff offset
        
        # Also find velocity direction changes
        if len(velocity_magnitudes) > 2:
            velocity_changes = np.diff(velocity_magnitudes)
            sign_changes = np.diff(np.sign(velocity_changes))
            
            for i, sign_change in enumerate(sign_changes):
                if abs(sign_change) > 1:  # Sign flip
                    frame_idx = i + 1
                    if frame_idx not in transition_points:
                        transition_points.append(frame_idx)
        
        return sorted(transition_points)
    
    def _calculate_movement_intensity_profile(self, pose_features: List[PoseFeatures]) -> np.ndarray:
        """Calculate movement intensity over time."""
        if len(pose_features) < 2:
            return np.array([0.0])
        
        intensity_profile = []
        
        for i in range(len(pose_features)):
            # Calculate intensity based on joint angle variations and movement
            pf = pose_features[i]
            
            # Joint angle contribution
            joint_intensity = 0.0
            for angle_name, angle_value in pf.joint_angles.items():
                # Deviation from neutral position (180 degrees for most joints)
                neutral_angle = 180.0 if 'elbow' in angle_name or 'knee' in angle_name else 0.0
                joint_intensity += abs(angle_value - neutral_angle) / 180.0
            
            joint_intensity /= len(pf.joint_angles)
            
            # Movement contribution (if not first frame)
            movement_intensity = 0.0
            if i > 0:
                prev_center = pose_features[i-1].center_of_mass
                curr_center = pf.center_of_mass
                movement_intensity = np.linalg.norm(np.array(curr_center) - np.array(prev_center))
            
            # Combined intensity
            total_intensity = 0.7 * joint_intensity + 0.3 * movement_intensity
            intensity_profile.append(total_intensity)
        
        return np.array(intensity_profile)
    
    def _calculate_spatial_distribution(self, pose_features: List[PoseFeatures]) -> Dict[str, float]:
        """Calculate movement distribution across different body regions."""
        if len(pose_features) < 2:
            return {"upper_body": 0.0, "lower_body": 0.0, "arms": 0.0, "legs": 0.0}
        
        # Define body region landmark indices
        upper_body_indices = [11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Shoulders, head, face
        lower_body_indices = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]  # Hips, legs, feet
        arm_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # Arms and hands
        leg_indices = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]  # Legs and feet
        
        region_movements = {"upper_body": [], "lower_body": [], "arms": [], "legs": []}
        
        for i in range(1, len(pose_features)):
            prev_landmarks = pose_features[i-1].landmarks
            curr_landmarks = pose_features[i].landmarks
            
            # Calculate movement for each region
            for region, indices in [
                ("upper_body", upper_body_indices),
                ("lower_body", lower_body_indices), 
                ("arms", arm_indices),
                ("legs", leg_indices)
            ]:
                region_movement = 0.0
                valid_points = 0
                
                for idx in indices:
                    if idx < len(prev_landmarks) and idx < len(curr_landmarks):
                        prev_point = prev_landmarks[idx, :2]
                        curr_point = curr_landmarks[idx, :2]
                        movement = np.linalg.norm(curr_point - prev_point)
                        region_movement += movement
                        valid_points += 1
                
                if valid_points > 0:
                    region_movements[region].append(region_movement / valid_points)
                else:
                    region_movements[region].append(0.0)
        
        # Calculate average movement for each region
        spatial_distribution = {}
        for region, movements in region_movements.items():
            spatial_distribution[region] = np.mean(movements) if movements else 0.0
        
        return spatial_distribution

    def _generate_pose_embedding(self, pose_features: List[PoseFeatures]) -> np.ndarray:
        """Generate 384-dimensional pose embedding from pose features."""
        if not pose_features:
            return np.zeros(384)
        
        # Extract all landmarks for statistical analysis
        all_landmarks = np.array([pf.landmarks for pf in pose_features])
        
        # 1. Core pose statistics (132 features)
        # Mean and std of landmark positions (33 landmarks * 2 coords * 2 stats = 132)
        landmark_xy = all_landmarks[:, :, :2]  # Only x, y coordinates
        landmark_means = np.mean(landmark_xy, axis=0).flatten()  # 66 features
        landmark_stds = np.std(landmark_xy, axis=0).flatten()    # 66 features
        
        # 2. Joint angle dynamics (20 features)
        angle_stats = []
        angle_names = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee', 'torso_lean']
        for angle_name in angle_names:
            angles = [pf.joint_angles.get(angle_name, 180.0) for pf in pose_features]
            angle_stats.extend([
                np.mean(angles), 
                np.std(angles),
                np.max(angles) - np.min(angles),  # Range
                np.median(angles)  # Median for robustness
            ])
        
        # 3. Movement trajectory features (24 features)
        centers = np.array([pf.center_of_mass for pf in pose_features])
        if len(centers) > 1:
            # Velocity and acceleration
            velocities = np.diff(centers, axis=0)
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)
            accelerations = np.diff(velocities, axis=0) if len(velocities) > 1 else np.array([0.0])
            acceleration_magnitudes = np.linalg.norm(accelerations, axis=1) if len(accelerations) > 0 else np.array([0.0])
            
            movement_stats = [
                # Position statistics
                np.mean(centers[:, 0]), np.std(centers[:, 0]), np.ptp(centers[:, 0]),  # X stats
                np.mean(centers[:, 1]), np.std(centers[:, 1]), np.ptp(centers[:, 1]),  # Y stats
                
                # Velocity statistics
                np.mean(velocity_magnitudes), np.std(velocity_magnitudes), 
                np.max(velocity_magnitudes), np.min(velocity_magnitudes),
                
                # Acceleration statistics
                np.mean(acceleration_magnitudes), np.std(acceleration_magnitudes),
                np.max(acceleration_magnitudes) if len(acceleration_magnitudes) > 0 else 0.0,
                np.min(acceleration_magnitudes) if len(acceleration_magnitudes) > 0 else 0.0,
                
                # Trajectory complexity
                self._calculate_trajectory_complexity(centers),
                self._calculate_movement_smoothness(velocity_magnitudes),
                
                # Directional features
                self._calculate_directional_consistency(velocities),
                self._calculate_movement_symmetry(centers),
                
                # Temporal features
                self._calculate_movement_rhythm(velocity_magnitudes),
                self._calculate_acceleration_patterns(acceleration_magnitudes),
                
                # Spatial coverage
                np.ptp(centers[:, 0]) * np.ptp(centers[:, 1]),  # Area coverage
                np.sqrt(np.sum(np.var(centers, axis=0)))  # Spatial variance
            ]
        else:
            movement_stats = [0.0] * 24
        
        # 4. Body region analysis (32 features)
        body_region_stats = self._calculate_body_region_features(pose_features)
        
        # 5. Pose stability and confidence (16 features)
        stability_stats = self._calculate_pose_stability_features(pose_features)
        
        # 6. Advanced geometric features (24 features)
        geometric_stats = self._calculate_geometric_features(pose_features)
        
        # 7. Temporal dynamics (20 features)
        temporal_stats = self._calculate_temporal_dynamics_features(pose_features)
        
        # 8. Coordination and symmetry (16 features)
        coordination_stats = self._calculate_coordination_features(pose_features)
        
        # Combine all features
        embedding_parts = [
            landmark_means,      # 66
            landmark_stds,       # 66
            angle_stats,         # 20
            movement_stats,      # 24
            body_region_stats,   # 32
            stability_stats,     # 16
            geometric_stats,     # 24
            temporal_stats,      # 20
            coordination_stats   # 16
        ]
        
        # Total: 66 + 66 + 20 + 24 + 32 + 16 + 24 + 20 + 16 = 284 features
        # Pad to exactly 384 dimensions
        embedding = np.concatenate(embedding_parts)
        
        if len(embedding) < 384:
            padding = np.zeros(384 - len(embedding))
            embedding = np.concatenate([embedding, padding])
        elif len(embedding) > 384:
            embedding = embedding[:384]
        
        return embedding
    
    def _generate_movement_embedding(self, movement_dynamics: MovementDynamics) -> np.ndarray:
        """Generate enhanced movement pattern embedding from dynamics."""
        # Create a comprehensive representation of movement patterns
        embedding_parts = [
            # Basic movement features
            movement_dynamics.spatial_coverage,
            movement_dynamics.rhythm_score,
            movement_dynamics.complexity_score,
            np.mean(movement_dynamics.velocity_profile) if len(movement_dynamics.velocity_profile) > 0 else 0.0,
            np.std(movement_dynamics.velocity_profile) if len(movement_dynamics.velocity_profile) > 0 else 0.0,
            np.mean(movement_dynamics.acceleration_profile) if len(movement_dynamics.acceleration_profile) > 0 else 0.0,
            np.std(movement_dynamics.acceleration_profile) if len(movement_dynamics.acceleration_profile) > 0 else 0.0,
            
            # Enhanced movement features
            movement_dynamics.footwork_area_coverage,
            movement_dynamics.upper_body_movement_range,
            movement_dynamics.rhythm_compatibility_score,
            movement_dynamics.movement_periodicity,
            len(movement_dynamics.transition_points) / max(1, len(movement_dynamics.velocity_profile)),  # Transition density
            np.mean(movement_dynamics.movement_intensity_profile) if len(movement_dynamics.movement_intensity_profile) > 0 else 0.0,
            np.std(movement_dynamics.movement_intensity_profile) if len(movement_dynamics.movement_intensity_profile) > 0 else 0.0,
            
            # Spatial distribution features
            movement_dynamics.spatial_distribution.get("upper_body", 0.0),
            movement_dynamics.spatial_distribution.get("lower_body", 0.0),
            movement_dynamics.spatial_distribution.get("arms", 0.0),
            movement_dynamics.spatial_distribution.get("legs", 0.0),
            
            # One-hot encoding for energy level
            1.0 if movement_dynamics.energy_level == "low" else 0.0,
            1.0 if movement_dynamics.energy_level == "medium" else 0.0,
            1.0 if movement_dynamics.energy_level == "high" else 0.0,
            
            # One-hot encoding for dominant direction
            1.0 if movement_dynamics.dominant_movement_direction == "horizontal" else 0.0,
            1.0 if movement_dynamics.dominant_movement_direction == "horizontal_left" else 0.0,
            1.0 if movement_dynamics.dominant_movement_direction == "vertical_up" else 0.0,
            1.0 if movement_dynamics.dominant_movement_direction == "vertical_down" else 0.0,
            1.0 if movement_dynamics.dominant_movement_direction == "static" else 0.0,
        ]
        
        embedding = np.array(embedding_parts)
        return embedding
    
    def _calculate_trajectory_complexity(self, centers: np.ndarray) -> float:
        """Calculate trajectory complexity using path length vs displacement ratio."""
        if len(centers) < 2:
            return 0.0
        
        # Calculate total path length
        path_length = np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1))
        
        # Calculate direct displacement
        displacement = np.linalg.norm(centers[-1] - centers[0])
        
        # Complexity ratio (higher = more complex path)
        if displacement < 1e-6:
            return 1.0 if path_length > 1e-6 else 0.0
        
        return min(path_length / displacement, 10.0) / 10.0  # Normalize to 0-1
    
    def _calculate_movement_smoothness(self, velocity_magnitudes: np.ndarray) -> float:
        """Calculate movement smoothness using velocity variation."""
        if len(velocity_magnitudes) < 2:
            return 1.0
        
        # Calculate jerk (derivative of acceleration)
        velocity_changes = np.diff(velocity_magnitudes)
        jerk = np.std(velocity_changes)
        
        # Convert to smoothness score (lower jerk = higher smoothness)
        return 1.0 / (1.0 + jerk * 100)
    
    def _calculate_directional_consistency(self, velocities: np.ndarray) -> float:
        """Calculate consistency of movement direction."""
        if len(velocities) < 2:
            return 1.0
        
        # Calculate angles between consecutive velocity vectors
        angles = []
        for i in range(len(velocities) - 1):
            v1, v2 = velocities[i], velocities[i + 1]
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        if not angles:
            return 1.0
        
        # Consistency is inverse of angular variation
        angular_std = np.std(angles)
        return 1.0 / (1.0 + angular_std)
    
    def _calculate_movement_symmetry(self, centers: np.ndarray) -> float:
        """Calculate bilateral movement symmetry."""
        if len(centers) < 2:
            return 1.0
        
        # Calculate center of trajectory
        trajectory_center = np.mean(centers, axis=0)
        
        # Calculate symmetry by comparing left and right deviations
        left_deviations = []
        right_deviations = []
        
        for center in centers:
            if center[0] < trajectory_center[0]:
                left_deviations.append(abs(center[0] - trajectory_center[0]))
            else:
                right_deviations.append(abs(center[0] - trajectory_center[0]))
        
        if not left_deviations or not right_deviations:
            return 0.5
        
        # Symmetry based on similar variance on both sides
        left_var = np.var(left_deviations)
        right_var = np.var(right_deviations)
        
        symmetry = 1.0 - abs(left_var - right_var) / (left_var + right_var + 1e-6)
        return max(0.0, symmetry)
    
    def _calculate_movement_rhythm(self, velocity_magnitudes: np.ndarray) -> float:
        """Calculate rhythmic consistency of movement."""
        if len(velocity_magnitudes) < 4:
            return 0.0
        
        # Use autocorrelation to find rhythmic patterns
        velocity_norm = velocity_magnitudes - np.mean(velocity_magnitudes)
        autocorr = np.correlate(velocity_norm, velocity_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        if len(autocorr) > 1:
            # Find secondary peaks indicating rhythm
            secondary_peaks = autocorr[1:min(len(autocorr), len(velocity_magnitudes)//2)]
            if len(secondary_peaks) > 0:
                return np.max(secondary_peaks) / (autocorr[0] + 1e-6)
        
        return 0.0
    
    def _calculate_acceleration_patterns(self, acceleration_magnitudes: np.ndarray) -> float:
        """Calculate acceleration pattern consistency."""
        if len(acceleration_magnitudes) < 2:
            return 0.0
        
        # Calculate coefficient of variation for acceleration
        mean_accel = np.mean(acceleration_magnitudes)
        std_accel = np.std(acceleration_magnitudes)
        
        if mean_accel < 1e-6:
            return 0.0
        
        # Lower coefficient of variation = more consistent patterns
        cv = std_accel / mean_accel
        return 1.0 / (1.0 + cv)
    
    def _calculate_body_region_features(self, pose_features: List[PoseFeatures]) -> List[float]:
        """Calculate features for different body regions (32 features)."""
        if not pose_features:
            return [0.0] * 32
        
        # Define body regions
        regions = {
            'head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'torso': [11, 12, 23, 24],
            'left_arm': [11, 13, 15, 17, 19, 21],
            'right_arm': [12, 14, 16, 18, 20, 22],
            'left_leg': [23, 25, 27, 29, 31],
            'right_leg': [24, 26, 28, 30, 32]
        }
        
        region_features = []
        
        for region_name, indices in regions.items():
            # Extract region landmarks across all frames
            region_landmarks = []
            for pf in pose_features:
                if pf.landmarks.shape[0] > max(indices):
                    region_points = pf.landmarks[indices, :2]  # x, y coordinates
                    region_landmarks.append(region_points.flatten())
            
            if region_landmarks:
                region_landmarks = np.array(region_landmarks)
                
                # Calculate region-specific features
                region_mean_movement = np.mean(np.std(region_landmarks, axis=0))
                region_max_movement = np.max(np.std(region_landmarks, axis=0))
                region_coordination = 1.0 - np.std(np.mean(region_landmarks, axis=1)) / (np.mean(np.mean(region_landmarks, axis=1)) + 1e-6)
                region_stability = 1.0 / (1.0 + np.mean(np.std(region_landmarks, axis=0)))
                
                region_features.extend([
                    region_mean_movement,
                    region_max_movement,
                    region_coordination,
                    region_stability
                ])
            else:
                region_features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Should be 6 regions * 4 features = 24, pad to 32
        while len(region_features) < 32:
            region_features.append(0.0)
        
        return region_features[:32]
    
    def _calculate_pose_stability_features(self, pose_features: List[PoseFeatures]) -> List[float]:
        """Calculate pose stability and confidence features (16 features)."""
        if not pose_features:
            return [0.0] * 16
        
        # Confidence statistics
        confidences = [pf.confidence for pf in pose_features]
        conf_mean = np.mean(confidences)
        conf_std = np.std(confidences)
        conf_min = np.min(confidences)
        conf_max = np.max(confidences)
        
        # Bounding box stability
        bboxes = np.array([pf.bounding_box for pf in pose_features])
        bbox_width_std = np.std(bboxes[:, 2])
        bbox_height_std = np.std(bboxes[:, 3])
        bbox_center_std = np.std(bboxes[:, :2], axis=0)
        
        # Joint angle stability
        angle_stabilities = []
        for angle_name in ['left_elbow', 'right_elbow', 'left_knee', 'right_knee', 'torso_lean']:
            angles = [pf.joint_angles.get(angle_name, 180.0) for pf in pose_features]
            angle_stabilities.append(1.0 / (1.0 + np.std(angles) / 180.0))
        
        # Landmark visibility consistency
        all_landmarks = np.array([pf.landmarks for pf in pose_features])
        visibility_consistency = np.mean(np.std(all_landmarks[:, :, 3], axis=0))  # Visibility scores
        
        stability_features = [
            conf_mean, conf_std, conf_min, conf_max,
            bbox_width_std, bbox_height_std,
            bbox_center_std[0], bbox_center_std[1],
            np.mean(angle_stabilities),
            np.std(angle_stabilities),
            visibility_consistency,
            1.0 / (1.0 + visibility_consistency),  # Stability score
            np.mean([pf.confidence for pf in pose_features[-3:]]) if len(pose_features) >= 3 else conf_mean,  # Recent confidence
            np.mean([pf.confidence for pf in pose_features[:3]]) if len(pose_features) >= 3 else conf_mean,   # Initial confidence
            0.0,  # Reserved
            0.0   # Reserved
        ]
        
        return stability_features
    
    def _calculate_geometric_features(self, pose_features: List[PoseFeatures]) -> List[float]:
        """Calculate geometric relationship features (24 features)."""
        if not pose_features:
            return [0.0] * 24
        
        geometric_features = []
        
        for pf in pose_features:
            landmarks = pf.landmarks
            if landmarks.shape[0] < 33:
                continue
            
            # Calculate key distances and ratios
            # Shoulder width
            shoulder_width = np.linalg.norm(landmarks[11, :2] - landmarks[12, :2])
            
            # Hip width
            hip_width = np.linalg.norm(landmarks[23, :2] - landmarks[24, :2])
            
            # Torso height (shoulder to hip)
            torso_height = np.linalg.norm(
                (landmarks[11, :2] + landmarks[12, :2]) / 2 - 
                (landmarks[23, :2] + landmarks[24, :2]) / 2
            )
            
            # Arm spans
            left_arm_span = (np.linalg.norm(landmarks[11, :2] - landmarks[13, :2]) + 
                           np.linalg.norm(landmarks[13, :2] - landmarks[15, :2]))
            right_arm_span = (np.linalg.norm(landmarks[12, :2] - landmarks[14, :2]) + 
                            np.linalg.norm(landmarks[14, :2] - landmarks[16, :2]))
            
            # Leg lengths
            left_leg_length = (np.linalg.norm(landmarks[23, :2] - landmarks[25, :2]) + 
                             np.linalg.norm(landmarks[25, :2] - landmarks[27, :2]))
            right_leg_length = (np.linalg.norm(landmarks[24, :2] - landmarks[26, :2]) + 
                              np.linalg.norm(landmarks[26, :2] - landmarks[28, :2]))
            
            geometric_features.extend([
                shoulder_width, hip_width, torso_height,
                left_arm_span, right_arm_span,
                left_leg_length, right_leg_length,
                shoulder_width / (hip_width + 1e-6),  # Shoulder-hip ratio
                (left_arm_span + right_arm_span) / (2 * torso_height + 1e-6),  # Arm-torso ratio
                abs(left_arm_span - right_arm_span) / (left_arm_span + right_arm_span + 1e-6),  # Arm asymmetry
                abs(left_leg_length - right_leg_length) / (left_leg_length + right_leg_length + 1e-6),  # Leg asymmetry
                pf.bounding_box[2] / (pf.bounding_box[3] + 1e-6)  # Aspect ratio
            ])
            break  # Use first valid frame for geometric baseline
        
        # Pad to 24 features if needed
        while len(geometric_features) < 24:
            geometric_features.append(0.0)
        
        return geometric_features[:24]
    
    def _calculate_temporal_dynamics_features(self, pose_features: List[PoseFeatures]) -> List[float]:
        """Calculate temporal dynamics features (20 features)."""
        if len(pose_features) < 2:
            return [0.0] * 20
        
        # Extract time series of key measurements
        confidences = [pf.confidence for pf in pose_features]
        centers = np.array([pf.center_of_mass for pf in pose_features])
        
        # Joint angle time series
        angle_series = {}
        for angle_name in ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']:
            angle_series[angle_name] = [pf.joint_angles.get(angle_name, 180.0) for pf in pose_features]
        
        temporal_features = []
        
        # Confidence dynamics
        conf_trend = np.polyfit(range(len(confidences)), confidences, 1)[0] if len(confidences) > 1 else 0.0
        conf_periodicity = self._calculate_movement_rhythm(np.array(confidences))
        
        # Movement dynamics
        if len(centers) > 1:
            velocities = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            vel_trend = np.polyfit(range(len(velocities)), velocities, 1)[0] if len(velocities) > 1 else 0.0
            vel_periodicity = self._calculate_movement_rhythm(velocities)
        else:
            vel_trend = 0.0
            vel_periodicity = 0.0
        
        # Joint angle dynamics
        angle_trends = []
        angle_periodicities = []
        for angle_name, angles in angle_series.items():
            if len(angles) > 1:
                trend = np.polyfit(range(len(angles)), angles, 1)[0]
                periodicity = self._calculate_movement_rhythm(np.array(angles))
            else:
                trend = 0.0
                periodicity = 0.0
            angle_trends.append(trend)
            angle_periodicities.append(periodicity)
        
        temporal_features = [
            conf_trend, conf_periodicity,
            vel_trend, vel_periodicity,
            *angle_trends,      # 4 features
            *angle_periodicities,  # 4 features
            np.mean(angle_trends),
            np.std(angle_trends),
            np.mean(angle_periodicities),
            np.std(angle_periodicities),
            len(pose_features) / 30.0,  # Duration in seconds (assuming 30 fps)
            np.std(confidences),
            0.0,  # Reserved
            0.0   # Reserved
        ]
        
        return temporal_features[:20]
    
    def _calculate_coordination_features(self, pose_features: List[PoseFeatures]) -> List[float]:
        """Calculate coordination and symmetry features (16 features)."""
        if len(pose_features) < 2:
            return [0.0] * 16
        
        coordination_features = []
        
        # Left-right symmetry in joint angles
        left_right_correlations = []
        angle_pairs = [('left_elbow', 'right_elbow'), ('left_knee', 'right_knee')]
        
        for left_angle, right_angle in angle_pairs:
            left_angles = [pf.joint_angles.get(left_angle, 180.0) for pf in pose_features]
            right_angles = [pf.joint_angles.get(right_angle, 180.0) for pf in pose_features]
            
            if len(left_angles) > 1 and len(right_angles) > 1:
                correlation = np.corrcoef(left_angles, right_angles)[0, 1]
                if not np.isnan(correlation):
                    left_right_correlations.append(abs(correlation))
                else:
                    left_right_correlations.append(0.0)
            else:
                left_right_correlations.append(0.0)
        
        # Upper-lower body coordination
        upper_body_movement = []
        lower_body_movement = []
        
        for i in range(1, len(pose_features)):
            prev_landmarks = pose_features[i-1].landmarks
            curr_landmarks = pose_features[i].landmarks
            
            # Upper body movement (shoulders, elbows, wrists)
            upper_indices = [11, 12, 13, 14, 15, 16]
            upper_movement = 0.0
            for idx in upper_indices:
                if idx < len(prev_landmarks) and idx < len(curr_landmarks):
                    movement = np.linalg.norm(curr_landmarks[idx, :2] - prev_landmarks[idx, :2])
                    upper_movement += movement
            upper_body_movement.append(upper_movement / len(upper_indices))
            
            # Lower body movement (hips, knees, ankles)
            lower_indices = [23, 24, 25, 26, 27, 28]
            lower_movement = 0.0
            for idx in lower_indices:
                if idx < len(prev_landmarks) and idx < len(curr_landmarks):
                    movement = np.linalg.norm(curr_landmarks[idx, :2] - prev_landmarks[idx, :2])
                    lower_movement += movement
            lower_body_movement.append(lower_movement / len(lower_indices))
        
        # Calculate upper-lower coordination
        if len(upper_body_movement) > 1 and len(lower_body_movement) > 1:
            upper_lower_correlation = np.corrcoef(upper_body_movement, lower_body_movement)[0, 1]
            if np.isnan(upper_lower_correlation):
                upper_lower_correlation = 0.0
        else:
            upper_lower_correlation = 0.0
        
        # Movement phase relationships
        centers = np.array([pf.center_of_mass for pf in pose_features])
        if len(centers) > 2:
            x_movement = centers[:, 0] - np.mean(centers[:, 0])
            y_movement = centers[:, 1] - np.mean(centers[:, 1])
            
            # Phase relationship between x and y movement
            if len(x_movement) > 1 and len(y_movement) > 1:
                xy_correlation = np.corrcoef(x_movement, y_movement)[0, 1]
                if np.isnan(xy_correlation):
                    xy_correlation = 0.0
            else:
                xy_correlation = 0.0
        else:
            xy_correlation = 0.0
        
        coordination_features = [
            *left_right_correlations,  # 2 features
            upper_lower_correlation,
            xy_correlation,
            np.mean(upper_body_movement),
            np.std(upper_body_movement),
            np.mean(lower_body_movement),
            np.std(lower_body_movement),
            np.mean(upper_body_movement) / (np.mean(lower_body_movement) + 1e-6),  # Upper/lower ratio
            abs(np.mean(upper_body_movement) - np.mean(lower_body_movement)),  # Upper-lower difference
            np.std(left_right_correlations) if left_right_correlations else 0.0,
            max(left_right_correlations) if left_right_correlations else 0.0,
            min(left_right_correlations) if left_right_correlations else 0.0,
            0.0,  # Reserved
            0.0,  # Reserved
            0.0   # Reserved
        ]
        
        return coordination_features[:16]

    def _calculate_analysis_quality(self, pose_features: List[PoseFeatures], 
                                  hand_features: List[HandFeatures]) -> float:
        """Calculate overall analysis quality score."""
        if not pose_features:
            return 0.0
        
        # Pose detection quality
        avg_pose_confidence = np.mean([pf.confidence for pf in pose_features])
        
        # Hand detection quality
        hand_detection_rate = sum(
            1 for hf in hand_features 
            if hf.left_hand_landmarks is not None or hf.right_hand_landmarks is not None
        ) / len(hand_features)
        
        # Combined quality score
        quality = 0.7 * avg_pose_confidence + 0.3 * hand_detection_rate
        
        return quality
    
    def calculate_movement_complexity_score(self, pose_features: List[PoseFeatures], 
                                          movement_dynamics: MovementDynamics) -> float:
        """
        Calculate movement complexity score based on joint angle variations and spatial coverage.
        Returns a score between 0.0 (simple) and 1.0 (complex).
        """
        if not pose_features:
            return 0.0
        
        complexity_components = []
        
        # 1. Joint angle variation complexity (0-1)
        joint_angle_complexity = self._calculate_joint_angle_complexity(pose_features)
        complexity_components.append(joint_angle_complexity)
        
        # 2. Spatial coverage complexity (0-1)
        spatial_complexity = min(movement_dynamics.spatial_coverage * 10, 1.0)  # Scale and cap at 1.0
        complexity_components.append(spatial_complexity)
        
        # 3. Movement velocity variation complexity (0-1)
        velocity_complexity = self._calculate_velocity_complexity(movement_dynamics.velocity_profile)
        complexity_components.append(velocity_complexity)
        
        # 4. Multi-limb coordination complexity (0-1)
        coordination_complexity = self._calculate_coordination_complexity(pose_features)
        complexity_components.append(coordination_complexity)
        
        # 5. Transition complexity (0-1)
        transition_complexity = len(movement_dynamics.transition_points) / max(1, len(pose_features)) * 2
        transition_complexity = min(transition_complexity, 1.0)
        complexity_components.append(transition_complexity)
        
        # 6. Rhythm complexity (0-1)
        rhythm_complexity = 1.0 - movement_dynamics.rhythm_score  # Less rhythmic = more complex
        complexity_components.append(rhythm_complexity)
        
        # Weighted combination of complexity components
        weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
        complexity_score = sum(w * c for w, c in zip(weights, complexity_components))
        
        return max(0.0, min(1.0, complexity_score))
    
    def _calculate_joint_angle_complexity(self, pose_features: List[PoseFeatures]) -> float:
        """Calculate complexity based on joint angle variations."""
        if len(pose_features) < 2:
            return 0.0
        
        angle_variations = []
        angle_names = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee', 'torso_lean']
        
        for angle_name in angle_names:
            angles = [pf.joint_angles.get(angle_name, 180.0) for pf in pose_features]
            if len(angles) > 1:
                # Calculate variation as coefficient of variation
                mean_angle = np.mean(angles)
                std_angle = np.std(angles)
                if mean_angle > 0:
                    cv = std_angle / mean_angle
                    angle_variations.append(min(cv, 1.0))  # Cap at 1.0
                else:
                    angle_variations.append(std_angle / 180.0)  # Normalize by max possible angle
        
        return np.mean(angle_variations) if angle_variations else 0.0
    
    def _calculate_velocity_complexity(self, velocity_profile: np.ndarray) -> float:
        """Calculate complexity based on velocity variations."""
        if len(velocity_profile) < 2:
            return 0.0
        
        # Calculate velocity variation metrics
        velocity_std = np.std(velocity_profile)
        velocity_mean = np.mean(velocity_profile)
        
        # Coefficient of variation
        if velocity_mean > 1e-6:
            cv = velocity_std / velocity_mean
        else:
            cv = velocity_std * 100  # Scale for very small movements
        
        # Number of direction changes
        velocity_changes = np.diff(velocity_profile)
        direction_changes = np.sum(np.diff(np.sign(velocity_changes)) != 0)
        direction_change_rate = direction_changes / len(velocity_profile)
        
        # Combine metrics
        complexity = 0.7 * min(cv, 1.0) + 0.3 * min(direction_change_rate * 2, 1.0)
        return complexity
    
    def _calculate_coordination_complexity(self, pose_features: List[PoseFeatures]) -> float:
        """Calculate complexity based on multi-limb coordination requirements."""
        if len(pose_features) < 2:
            return 0.0
        
        # Calculate independent movement of different body parts
        body_parts = {
            'left_arm': [11, 13, 15],    # shoulder, elbow, wrist
            'right_arm': [12, 14, 16],   # shoulder, elbow, wrist
            'left_leg': [23, 25, 27],    # hip, knee, ankle
            'right_leg': [24, 26, 28],   # hip, knee, ankle
            'torso': [11, 12, 23, 24]    # shoulders and hips
        }
        
        part_movements = {}
        
        for part_name, indices in body_parts.items():
            movements = []
            for i in range(1, len(pose_features)):
                prev_landmarks = pose_features[i-1].landmarks
                curr_landmarks = pose_features[i].landmarks
                
                part_movement = 0.0
                valid_points = 0
                
                for idx in indices:
                    if idx < len(prev_landmarks) and idx < len(curr_landmarks):
                        movement = np.linalg.norm(curr_landmarks[idx, :2] - prev_landmarks[idx, :2])
                        part_movement += movement
                        valid_points += 1
                
                if valid_points > 0:
                    movements.append(part_movement / valid_points)
                else:
                    movements.append(0.0)
            
            part_movements[part_name] = movements
        
        # Calculate independence of movement (low correlation = high complexity)
        correlations = []
        part_names = list(part_movements.keys())
        
        for i in range(len(part_names)):
            for j in range(i + 1, len(part_names)):
                movements1 = part_movements[part_names[i]]
                movements2 = part_movements[part_names[j]]
                
                if len(movements1) > 1 and len(movements2) > 1:
                    correlation = np.corrcoef(movements1, movements2)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))
        
        if correlations:
            # Lower average correlation = higher complexity
            avg_correlation = np.mean(correlations)
            coordination_complexity = 1.0 - avg_correlation
        else:
            coordination_complexity = 0.0
        
        return max(0.0, coordination_complexity)
    
    def calculate_tempo_compatibility_range(self, movement_dynamics: MovementDynamics, 
                                          pose_features: List[PoseFeatures]) -> Tuple[float, float]:
        """
        Calculate tempo compatibility range for the move based on movement analysis.
        Returns (min_bpm, max_bpm) tuple.
        """
        if not pose_features or len(pose_features) < 2:
            return (100.0, 130.0)  # Default Bachata range
        
        # Base Bachata tempo range
        base_min_bpm = 90.0
        base_max_bpm = 150.0
        
        # Analyze movement characteristics to refine tempo range
        
        # 1. Movement speed analysis
        avg_velocity = np.mean(movement_dynamics.velocity_profile) if len(movement_dynamics.velocity_profile) > 0 else 0.0
        
        # 2. Rhythm compatibility score
        rhythm_score = movement_dynamics.rhythm_compatibility_score
        
        # 3. Movement periodicity
        periodicity = movement_dynamics.movement_periodicity
        
        # 4. Energy level
        energy_level = movement_dynamics.energy_level
        
        # 5. Complexity score
        complexity = movement_dynamics.complexity_score
        
        # Adjust tempo range based on characteristics
        tempo_adjustments = []
        
        # Speed-based adjustment
        if avg_velocity < 0.01:  # Very slow movement
            tempo_adjustments.append((-20, -10))  # Prefer slower tempos
        elif avg_velocity > 0.05:  # Fast movement
            tempo_adjustments.append((10, 20))   # Prefer faster tempos
        
        # Energy-based adjustment
        if energy_level == "low":
            tempo_adjustments.append((-15, -5))
        elif energy_level == "high":
            tempo_adjustments.append((5, 15))
        
        # Complexity-based adjustment
        if complexity < 0.3:  # Simple moves
            tempo_adjustments.append((-10, 10))  # More flexible
        elif complexity > 0.7:  # Complex moves
            tempo_adjustments.append((-5, 5))    # Narrower range
        
        # Rhythm-based adjustment
        if rhythm_score > 0.7:  # Highly rhythmic
            tempo_adjustments.append((-5, 5))    # Narrower range for precision
        elif rhythm_score < 0.3:  # Less rhythmic
            tempo_adjustments.append((-15, 15))  # Wider range for flexibility
        
        # Apply adjustments
        min_adjustment = sum(adj[0] for adj in tempo_adjustments)
        max_adjustment = sum(adj[1] for adj in tempo_adjustments)
        
        final_min_bpm = max(base_min_bpm + min_adjustment, 80.0)   # Hard minimum
        final_max_bpm = min(base_max_bpm + max_adjustment, 160.0)  # Hard maximum
        
        # Ensure min < max
        if final_min_bpm >= final_max_bpm:
            final_min_bpm = final_max_bpm - 10.0
        
        return (final_min_bpm, final_max_bpm)
    
    def calculate_difficulty_score(self, movement_dynamics: MovementDynamics, 
                                 pose_features: List[PoseFeatures]) -> float:
        """
        Calculate difficulty score using movement speed, complexity, and coordination requirements.
        Returns a score between 0.0 (beginner) and 1.0 (advanced).
        """
        if not pose_features:
            return 0.0
        
        difficulty_components = []
        
        # 1. Movement speed difficulty (0-1)
        avg_velocity = np.mean(movement_dynamics.velocity_profile) if len(movement_dynamics.velocity_profile) > 0 else 0.0
        speed_difficulty = min(avg_velocity * 20, 1.0)  # Scale velocity to 0-1
        difficulty_components.append(speed_difficulty)
        
        # 2. Complexity difficulty (0-1)
        complexity_difficulty = self.calculate_movement_complexity_score(pose_features, movement_dynamics)
        difficulty_components.append(complexity_difficulty)
        
        # 3. Coordination difficulty (0-1)
        coordination_difficulty = self._calculate_coordination_difficulty(pose_features)
        difficulty_components.append(coordination_difficulty)
        
        # 4. Balance and stability difficulty (0-1)
        stability_difficulty = self._calculate_stability_difficulty(pose_features, movement_dynamics)
        difficulty_components.append(stability_difficulty)
        
        # 5. Rhythm precision difficulty (0-1)
        rhythm_difficulty = 1.0 - movement_dynamics.rhythm_score  # Less rhythmic = more difficult
        difficulty_components.append(rhythm_difficulty)
        
        # 6. Spatial coverage difficulty (0-1)
        spatial_difficulty = min(movement_dynamics.spatial_coverage * 5, 1.0)
        difficulty_components.append(spatial_difficulty)
        
        # 7. Transition difficulty (0-1)
        transition_density = len(movement_dynamics.transition_points) / max(1, len(pose_features))
        transition_difficulty = min(transition_density * 3, 1.0)
        difficulty_components.append(transition_difficulty)
        
        # Weighted combination
        weights = [0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10]
        difficulty_score = sum(w * d for w, d in zip(weights, difficulty_components))
        
        return max(0.0, min(1.0, difficulty_score))
    
    def _calculate_coordination_difficulty(self, pose_features: List[PoseFeatures]) -> float:
        """Calculate difficulty based on coordination requirements."""
        if len(pose_features) < 2:
            return 0.0
        
        # Calculate simultaneous movement of multiple body parts
        simultaneous_movements = []
        
        for i in range(1, len(pose_features)):
            prev_landmarks = pose_features[i-1].landmarks
            curr_landmarks = pose_features[i].landmarks
            
            # Count how many body parts are moving simultaneously
            moving_parts = 0
            body_parts = {
                'left_arm': [11, 13, 15],
                'right_arm': [12, 14, 16],
                'left_leg': [23, 25, 27],
                'right_leg': [24, 26, 28],
                'torso': [0, 11, 12]  # head and shoulders
            }
            
            movement_threshold = 0.01  # Minimum movement to consider "moving"
            
            for part_name, indices in body_parts.items():
                part_movement = 0.0
                valid_points = 0
                
                for idx in indices:
                    if idx < len(prev_landmarks) and idx < len(curr_landmarks):
                        movement = np.linalg.norm(curr_landmarks[idx, :2] - prev_landmarks[idx, :2])
                        part_movement += movement
                        valid_points += 1
                
                if valid_points > 0:
                    avg_movement = part_movement / valid_points
                    if avg_movement > movement_threshold:
                        moving_parts += 1
            
            simultaneous_movements.append(moving_parts)
        
        # Difficulty increases with more simultaneous movements
        avg_simultaneous = np.mean(simultaneous_movements)
        max_simultaneous = max(simultaneous_movements) if simultaneous_movements else 0
        
        # Normalize to 0-1 (5 body parts max)
        coordination_difficulty = (0.6 * avg_simultaneous + 0.4 * max_simultaneous) / 5.0
        
        return min(coordination_difficulty, 1.0)
    
    def _calculate_stability_difficulty(self, pose_features: List[PoseFeatures], 
                                      movement_dynamics: MovementDynamics) -> float:
        """Calculate difficulty based on balance and stability requirements."""
        if not pose_features:
            return 0.0
        
        stability_factors = []
        
        # 1. Center of mass variation
        centers = np.array([pf.center_of_mass for pf in pose_features])
        if len(centers) > 1:
            center_variation = np.std(centers, axis=0)
            center_instability = np.linalg.norm(center_variation)
            stability_factors.append(min(center_instability * 10, 1.0))
        
        # 2. Base of support analysis (foot positioning)
        foot_positions = []
        for pf in pose_features:
            landmarks = pf.landmarks
            if landmarks.shape[0] > 32:
                # Use ankle positions as proxy for base of support
                left_ankle = landmarks[27, :2]
                right_ankle = landmarks[28, :2]
                foot_distance = np.linalg.norm(left_ankle - right_ankle)
                foot_positions.append(foot_distance)
        
        if foot_positions:
            foot_variation = np.std(foot_positions)
            base_instability = min(foot_variation * 5, 1.0)
            stability_factors.append(base_instability)
        
        # 3. Acceleration-based instability
        if len(movement_dynamics.acceleration_profile) > 0:
            accel_variation = np.std(movement_dynamics.acceleration_profile)
            accel_instability = min(accel_variation * 2, 1.0)
            stability_factors.append(accel_instability)
        
        # 4. Confidence variation (lower confidence = less stable poses)
        confidences = [pf.confidence for pf in pose_features]
        confidence_variation = np.std(confidences)
        confidence_instability = min(confidence_variation * 2, 1.0)
        stability_factors.append(confidence_instability)
        
        return np.mean(stability_factors) if stability_factors else 0.0

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'pose_detector'):
            self.pose_detector.close()
        if hasattr(self, 'hand_detector'):
            self.hand_detector.close()


# Utility functions for batch processing
def analyze_video_directory(directory_path: str, 
                          analyzer: MoveAnalyzer,
                          file_pattern: str = "*.mp4") -> Dict[str, MoveAnalysisResult]:
    """
    Analyze all video files in a directory.
    
    Args:
        directory_path: Path to directory containing video files
        analyzer: MoveAnalyzer instance
        file_pattern: File pattern to match (default: "*.mp4")
        
    Returns:
        Dictionary mapping video filenames to analysis results
    """
    directory = Path(directory_path)
    video_files = list(directory.glob(file_pattern))
    
    results = {}
    
    logger.info(f"Found {len(video_files)} video files in {directory_path}")
    
    for video_file in tqdm(video_files, desc="Analyzing videos"):
        try:
            result = analyzer.analyze_move_clip(str(video_file))
            results[video_file.name] = result
            logger.info(f"Successfully analyzed {video_file.name}")
        except Exception as e:
            logger.error(f"Failed to analyze {video_file.name}: {e}")
    
    return results


def save_analysis_results(results: Dict[str, MoveAnalysisResult], 
                         output_path: str) -> None:
    """
    Save analysis results to a file.
    
    Args:
        results: Dictionary of analysis results
        output_path: Path to save results
    """
    import pickle
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Saved analysis results to {output_path}")


def load_analysis_results(input_path: str) -> Dict[str, MoveAnalysisResult]:
    """
    Load analysis results from a file.
    
    Args:
        input_path: Path to load results from
        
    Returns:
        Dictionary of analysis results
    """
    import pickle
    
    with open(input_path, 'rb') as f:
        results = pickle.load(f)
    
    logger.info(f"Loaded analysis results from {input_path}")
    return results


def calculate_transition_compatibility(move1_result: MoveAnalysisResult, 
                                     move2_result: MoveAnalysisResult) -> float:
    """
    Calculate transition compatibility between two moves.
    
    Args:
        move1_result: Analysis result for the first move
        move2_result: Analysis result for the second move
        
    Returns:
        Compatibility score between 0.0 and 1.0
    """
    if not move1_result.pose_features or not move2_result.pose_features:
        return 0.0
    
    # Get ending pose of first move and starting pose of second move
    end_pose1 = move1_result.pose_features[-1]
    start_pose2 = move2_result.pose_features[0]
    
    # Calculate pose similarity
    pose_similarity = _calculate_pose_similarity(end_pose1, start_pose2)
    
    # Calculate energy level compatibility
    energy1 = move1_result.movement_dynamics.energy_level
    energy2 = move2_result.movement_dynamics.energy_level
    energy_compatibility = _calculate_energy_compatibility(energy1, energy2)
    
    # Calculate movement direction compatibility
    dir1 = move1_result.movement_dynamics.dominant_movement_direction
    dir2 = move2_result.movement_dynamics.dominant_movement_direction
    direction_compatibility = _calculate_direction_compatibility(dir1, dir2)
    
    # Calculate rhythm compatibility
    rhythm1 = move1_result.movement_dynamics.rhythm_compatibility_score
    rhythm2 = move2_result.movement_dynamics.rhythm_compatibility_score
    rhythm_compatibility = 1.0 - abs(rhythm1 - rhythm2)  # Higher when similar
    
    # Calculate complexity compatibility
    complexity1 = move1_result.movement_dynamics.complexity_score
    complexity2 = move2_result.movement_dynamics.complexity_score
    complexity_compatibility = 1.0 - abs(complexity1 - complexity2)
    
    # Weighted combination
    compatibility = (
        0.3 * pose_similarity +
        0.2 * energy_compatibility +
        0.2 * direction_compatibility +
        0.15 * rhythm_compatibility +
        0.15 * complexity_compatibility
    )
    
    return max(0.0, min(1.0, compatibility))


def _calculate_pose_similarity(pose1: PoseFeatures, pose2: PoseFeatures) -> float:
    """Calculate similarity between two poses."""
    # Compare joint angles
    angle_similarities = []
    
    for angle_name in ['left_elbow', 'right_elbow', 'left_knee', 'right_knee', 'torso_lean']:
        angle1 = pose1.joint_angles.get(angle_name, 180.0)
        angle2 = pose2.joint_angles.get(angle_name, 180.0)
        
        # Calculate angular similarity (closer angles = higher similarity)
        angle_diff = abs(angle1 - angle2)
        angle_similarity = 1.0 - (angle_diff / 180.0)  # Normalize to 0-1
        angle_similarities.append(max(0.0, angle_similarity))
    
    # Compare center of mass positions
    center1 = np.array(pose1.center_of_mass)
    center2 = np.array(pose2.center_of_mass)
    center_distance = np.linalg.norm(center1 - center2)
    center_similarity = 1.0 / (1.0 + center_distance * 10)  # Scale factor for normalization
    
    # Combine similarities
    pose_similarity = 0.8 * np.mean(angle_similarities) + 0.2 * center_similarity
    
    return pose_similarity


def _calculate_energy_compatibility(energy1: str, energy2: str) -> float:
    """Calculate compatibility between energy levels."""
    energy_levels = {"low": 0, "medium": 1, "high": 2}
    
    level1 = energy_levels.get(energy1, 1)
    level2 = energy_levels.get(energy2, 1)
    
    # Perfect match = 1.0, adjacent levels = 0.7, opposite levels = 0.3
    diff = abs(level1 - level2)
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.7
    else:
        return 0.3


def _calculate_direction_compatibility(dir1: str, dir2: str) -> float:
    """Calculate compatibility between movement directions."""
    if dir1 == dir2:
        return 1.0
    
    # Define direction compatibility matrix
    direction_compatibility_matrix = {
        ("horizontal", "horizontal_left"): 0.8,
        ("horizontal_left", "horizontal"): 0.8,
        ("vertical_up", "vertical_down"): 0.6,
        ("vertical_down", "vertical_up"): 0.6,
        ("static", "horizontal"): 0.9,
        ("static", "horizontal_left"): 0.9,
        ("static", "vertical_up"): 0.9,
        ("static", "vertical_down"): 0.9,
        ("horizontal", "static"): 0.9,
        ("horizontal_left", "static"): 0.9,
        ("vertical_up", "static"): 0.9,
        ("vertical_down", "static"): 0.9,
    }
    
    return direction_compatibility_matrix.get((dir1, dir2), 0.5)  # Default moderate compatibility


def analyze_move_transitions(results: Dict[str, MoveAnalysisResult]) -> Dict[Tuple[str, str], float]:
    """
    Analyze transition compatibility between all pairs of moves.
    
    Args:
        results: Dictionary of move analysis results
        
    Returns:
        Dictionary mapping move pairs to compatibility scores
    """
    transition_matrix = {}
    move_names = list(results.keys())
    
    logger.info(f"Calculating transition compatibility for {len(move_names)} moves")
    
    for i, move1 in enumerate(move_names):
        for j, move2 in enumerate(move_names):
            if i != j:  # Don't calculate self-transitions
                compatibility = calculate_transition_compatibility(
                    results[move1], results[move2]
                )
                transition_matrix[(move1, move2)] = compatibility
    
    logger.info(f"Calculated {len(transition_matrix)} transition compatibilities")
    return transition_matrix