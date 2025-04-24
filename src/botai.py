#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import base64
import requests
import json
import enum
import time
import os
from io import BytesIO
from PIL import Image as PILImage
from sensor_msgs.msg import Image, Range
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16, Float64, String
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion


# Define states as enum for clarity
class State(enum.Enum):
    FOLLOW_LINE = 1
    DETECT_OBJECT = 2
    APPROACH_OBJECT = 3
    GRAB_OBJECT = 4
    FIND_YELLOW_ZONE = 5
    GO_TO_YELLOW_ZONE = 6
    DISCARD_OBJECT = 7
    RETURN_TO_LINE = 8

# Define object types
class ObjectType(enum.Enum):
    UNKNOWN = 0
    RED = 1
    GREEN = 2
    BLUE = 3

class AIController:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('ai_controller')
        rospy.loginfo("AI Controller starting...")

        # Load environment variables and OpenAI API key
        self.api_key = ""
            rospy.logwarn("No OPENAI_API_KEY found in environment variables. Vision-based decision making will be disabled.")
            
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4-vision-preview"
        
        # ROS communication
        self.bridge = CvBridge()
        self.last_image = None
        self.last_depth_image = None
        self.current_pose = None
        self.start_pose = None
        self.fl_distance = float('inf')
        self.fr_distance = float('inf')
        
        # State machine
        self.state = State.FOLLOW_LINE
        self.object_type = ObjectType.UNKNOWN
        self.ai_decision_count = 0
        self.last_api_call_time = 0
        self.api_call_interval = 1.0  # 1 second between API calls
        
        # Parameters
        self.line_follow_speed = 0.1
        self.approach_speed = 0.08
        self.turn_speed = 0.3
        self.min_contour_area = 500
        self.gripper_open_position = 0
        self.gripper_close_position = 170
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.fl_sensor_sub = rospy.Subscriber('/range/fl', Range, self.fl_sensor_callback)
        self.fr_sensor_sub = rospy.Subscriber('/range/fr', Range, self.fr_sensor_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.servo_pub = rospy.Publisher('/servo', UInt16, queue_size=10)
        self.servo_load_pub = rospy.Publisher('/servoLoad', Float64, queue_size=10)
        self.state_pub = rospy.Publisher('/robot_state', String, queue_size=10)
        
        # Initialize twist
        self.twist = Twist()
        
        # Rate limiter
        self.rate = rospy.Rate(10)  # 10 Hz
        
        # Wait for services and setup to complete
        rospy.sleep(1)
        
        # Open gripper at start
        self.open_gripper()
        
        rospy.loginfo("AI Controller initialized and ready.")

    # Callback functions
    def image_callback(self, msg):
        try:
            self.last_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_state()
        except CvBridgeError as e:
            rospy.logerr(f"Image conversion error: {e}")
    
    def depth_callback(self, msg):
        try:
            self.last_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            rospy.logerr(f"Depth image conversion error: {e}")
    
    def fl_sensor_callback(self, msg):
        self.fl_distance = msg.range
    
    def fr_sensor_callback(self, msg):
        self.fr_distance = msg.range
    
    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.current_pose = (position.x, position.y, yaw)
        
        if self.start_pose is None:
            self.start_pose = self.current_pose
            rospy.loginfo(f"Recorded start pose: x={position.x:.2f}, y={position.y:.2f}, yaw={yaw:.2f}")
    
    # Robot control functions
    def stop_robot(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)
    
    def open_gripper(self):
        self.servo_pub.publish(UInt16(self.gripper_open_position))
        self.servo_load_pub.publish(Float64(0.5))
        rospy.loginfo("Gripper opened")
        rospy.sleep(1.0)
    
    def close_gripper(self):
        self.servo_pub.publish(UInt16(self.gripper_close_position))
        self.servo_load_pub.publish(Float64(0.8))
        rospy.loginfo("Gripper closed")
        rospy.sleep(1.0)
    
    def follow_line(self, line_mask, width):
        """Simple proportional line following"""
        # Find the centroid of the line
        M = cv2.moments(line_mask)
        if M["m00"] == 0:
            # No line detected, slow forward
            self.twist.linear.x = 0.05
            self.twist.angular.z = 0.0
            return
        
        cx = int(M["m10"] / M["m00"])
        error = cx - width // 2
        kp = 1.0 / 450.0  # Proportional gain
        
        self.twist.linear.x = self.line_follow_speed
        self.twist.angular.z = -kp * float(error)
        self.cmd_vel_pub.publish(self.twist)
    
    def process_image_for_line(self, image):
        """Process image to get line mask"""
        if image is None:
            return None
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w, _ = hsv.shape
        
        # Define black line thresholds
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 60])
        
        # Use bottom 20% of the image for line detection
        bot_start = int(h * 0.8)
        bot_end = h
        bot_roi = hsv[bot_start:bot_end, :]
        
        # Create mask for line
        line_mask = cv2.inRange(bot_roi, lower_black, upper_black)
        
        return line_mask
    
    def encode_image_for_api(self, image, max_width=512):
        """Encode and resize image for API call"""
        # Resize image to save bandwidth while preserving aspect ratio
        h, w = image.shape[:2]
        if w > max_width:
            new_h = int(h * (max_width / w))
            image = cv2.resize(image, (max_width, new_h))
        
        # Convert to RGB (from BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and encode to base64
        pil_img = PILImage.fromarray(image_rgb)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=70)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_str
    
    def call_openai_api(self, image, prompt):
        """Call OpenAI API with the image and prompt"""
        if not self.api_key:
            rospy.logerr("OpenAI API key not set in environment variables. Set OPENAI_API_KEY in .env file.")
            return None
        
        # Check rate limiting
        current_time = time.time()
        if current_time - self.last_api_call_time < self.api_call_interval:
            return None
        
        self.last_api_call_time = current_time
        self.ai_decision_count += 1
        
        # Encode the image
        base64_image = self.encode_image_for_api(image)
        
        # Prepare the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a vision-based robotics controller. You will receive images from a robot's camera and must make decisions based on what you see. Your responses should be concise and directly actionable by the robot system."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 100
        }
        
        rospy.loginfo(f"Calling OpenAI API (request #{self.ai_decision_count})...")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"].strip()
                rospy.loginfo(f"API response: {answer}")
                return answer
            else:
                rospy.logerr(f"Unexpected API response: {result}")
                return None
        except Exception as e:
            rospy.logerr(f"API call failed: {e}")
            return None
    
    def detect_object_type(self, image):
        """Use OpenAI API to detect the type of object in the image"""
        prompt = """
        Analyze this image and identify if there's a peg/object that the robot should pick up.
        If you see a colored peg, determine its color (RED, GREEN, or BLUE).
        Respond with EXACTLY ONE of these options:
        - RED_OBJECT
        - GREEN_OBJECT
        - BLUE_OBJECT
        - NO_OBJECT
        
        IMPORTANT: Your response must be just one of these four options, nothing else.
        """
        
        response = self.call_openai_api(image, prompt)
        
        if response == "RED_OBJECT":
            return ObjectType.RED
        elif response == "GREEN_OBJECT":
            return ObjectType.GREEN
        elif response == "BLUE_OBJECT":
            return ObjectType.BLUE
        else:
            return ObjectType.UNKNOWN
    
    def check_object_centered(self, image):
        """Use OpenAI API to check if the object is centered for grabbing"""
        prompt = """
        Analyze this image and determine if the colored object (peg) is properly centered for the robot's gripper.
        The robot's gripper is positioned at the center of the camera view.
        Respond with EXACTLY ONE of these options:
        - CENTERED (object is well-centered and ready to grab)
        - LEFT (object is to the left, robot should turn right)
        - RIGHT (object is to the right, robot should turn left)
        - FORWARD (object is centered but too far, robot should move forward)
        - NOT_VISIBLE (no object visible in frame)
        
        IMPORTANT: Your response must be just one of these five options, nothing else.
        """
        
        response = self.call_openai_api(image, prompt)
        return response
    
    def check_object_graspable(self, image):
        """Use OpenAI API to check if the object is ready to be grasped"""
        prompt = """
        Analyze this image and determine if the colored peg is ready to be grasped by the robot's gripper.
        The robot's gripper is positioned at the center bottom of the camera view.
        Respond with EXACTLY ONE of these options:
        - GRASP_NOW (object is in perfect position to grasp)
        - MOVE_CLOSER (object visible but too far away)
        - NOT_READY (object not in position to grasp)
        - NOT_VISIBLE (no object visible in frame)
        
        IMPORTANT: Your response must be just one of these four options, nothing else.
        """
        
        response = self.call_openai_api(image, prompt)
        return response
    
    def process_state(self):
        """Process the current state of the robot"""
        if self.last_image is None:
            return
        
        # Publish current state for monitoring
        self.state_pub.publish(String(self.state.name))
        
        # Display window with state information
        display_img = self.last_image.copy()
        cv2.putText(display_img, f"State: {self.state.name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.object_type != ObjectType.UNKNOWN:
            cv2.putText(display_img, f"Object: {self.object_type.name}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Resize for display
        display_img_resized = cv2.resize(display_img, (640, 480))
        cv2.imshow("Robot View", display_img_resized)
        cv2.waitKey(1)
        
        # State machine
        if self.state == State.FOLLOW_LINE:
            line_mask = self.process_image_for_line(self.last_image)
            if line_mask is not None:
                self.follow_line(line_mask, self.last_image.shape[1])
            
            # Periodically check for objects while following line
            if time.time() - self.last_api_call_time > 3.0:  # Check every 3 seconds
                detected_type = self.detect_object_type(self.last_image)
                if detected_type != ObjectType.UNKNOWN:
                    rospy.loginfo(f"Detected {detected_type.name} object while following line")
                    self.object_type = detected_type
                    self.state = State.DETECT_OBJECT
                    self.stop_robot()
        
        elif self.state == State.DETECT_OBJECT:
            # Confirm the object detection
            detected_type = self.detect_object_type(self.last_image)
            if detected_type != ObjectType.UNKNOWN:
                rospy.loginfo(f"Confirmed {detected_type.name} object")
                self.object_type = detected_type
                self.state = State.APPROACH_OBJECT
            else:
                # Go back to line following if object is not confirmed
                rospy.loginfo("Object not confirmed, returning to line following")
                self.state = State.FOLLOW_LINE
        
        elif self.state == State.APPROACH_OBJECT:
            # Check if object is centered and at the right distance
            position_status = self.check_object_centered(self.last_image)
            
            if position_status == "CENTERED":
                self.twist.linear.x = self.approach_speed
                self.twist.angular.z = 0.0
            elif position_status == "LEFT":
                self.twist.linear.x = 0.0
                self.twist.angular.z = -self.turn_speed
            elif position_status == "RIGHT":
                self.twist.linear.x = 0.0
                self.twist.angular.z = self.turn_speed
            elif position_status == "FORWARD":
                self.twist.linear.x = self.approach_speed
                self.twist.angular.z = 0.0
            elif position_status == "NOT_VISIBLE":
                # If object is lost, go back to line following
                rospy.loginfo("Object lost, returning to line following")
                self.state = State.FOLLOW_LINE
                self.object_type = ObjectType.UNKNOWN
            
            self.cmd_vel_pub.publish(self.twist)
            
            # Check if we're close enough to grab
            grasp_status = self.check_object_graspable(self.last_image)
            if grasp_status == "GRASP_NOW":
                self.stop_robot()
                self.state = State.GRAB_OBJECT
            
        elif self.state == State.GRAB_OBJECT:
            # Close gripper
            self.stop_robot()
            self.close_gripper()
            rospy.loginfo(f"Grabbed {self.object_type.name} object")
            
            # Decide what to do with the object
            if self.object_type == ObjectType.RED:
                self.state = State.FIND_YELLOW_ZONE
                rospy.loginfo("Moving RED object to yellow zone")
            else:
                self.state = State.DISCARD_OBJECT
                rospy.loginfo("Discarding object to the side")
        
        elif self.state == State.FIND_YELLOW_ZONE:
            # Simplified yellow zone finding - in a real implementation,
            # you would use the approach from yellow_follower.py with a 360 scan
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.3  # Turn to look for yellow zone
            self.cmd_vel_pub.publish(self.twist)
            
            # Use OpenAI to detect yellow zone
            prompt = """
            Analyze this image and determine if you can see a yellow zone or area.
            Respond with EXACTLY ONE of these options:
            - YELLOW_ZONE_VISIBLE
            - NO_YELLOW_ZONE
            
            IMPORTANT: Your response must be just one of these two options, nothing else.
            """
            
            response = self.call_openai_api(self.last_image, prompt)
            if response == "YELLOW_ZONE_VISIBLE":
                self.stop_robot()
                self.state = State.GO_TO_YELLOW_ZONE
        
        elif self.state == State.GO_TO_YELLOW_ZONE:
            # Move to the yellow zone
            prompt = """
            Analyze this image and determine if the robot should move towards the yellow zone.
            The yellow zone should appear as a yellow area or marking.
            Respond with EXACTLY ONE of these options:
            - MOVE_FORWARD (yellow zone is visible and ahead)
            - TURN_LEFT (yellow zone is to the left)
            - TURN_RIGHT (yellow zone is to the right)
            - REACHED_ZONE (robot has reached the yellow zone)
            - ZONE_NOT_VISIBLE (yellow zone is not visible)
            
            IMPORTANT: Your response must be just one of these five options, nothing else.
            """
            
            response = self.call_openai_api(self.last_image, prompt)
            
            if response == "MOVE_FORWARD":
                self.twist.linear.x = 0.1
                self.twist.angular.z = 0.0
            elif response == "TURN_LEFT":
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.3
            elif response == "TURN_RIGHT":
                self.twist.linear.x = 0.0
                self.twist.angular.z = -0.3
            elif response == "REACHED_ZONE":
                self.stop_robot()
                # Open gripper to release object
                self.open_gripper()
                rospy.loginfo("Released object in yellow zone")
                self.state = State.RETURN_TO_LINE
            elif response == "ZONE_NOT_VISIBLE":
                # Continue searching
                self.state = State.FIND_YELLOW_ZONE
            
            self.cmd_vel_pub.publish(self.twist)
        
        elif self.state == State.DISCARD_OBJECT:
            # Execute discard routine like in discard_peg2.py
            
            # Turn 90 degrees right
            self.twist.linear.x = 0.0
            self.twist.angular.z = -0.5
            self.cmd_vel_pub.publish(self.twist)
            rospy.sleep(3.2)
            
            # Move forward
            self.stop_robot()
            self.twist.linear.x = 0.1
            self.twist.angular.z = 0.0
            self.cmd_vel_pub.publish(self.twist)
            rospy.sleep(3.5)
            
            # Open gripper to release object
            self.stop_robot()
            self.open_gripper()
            rospy.loginfo("Released object in discard zone")
            
            # Move backward
            self.twist.linear.x = -0.1
            self.twist.angular.z = 0.0
            self.cmd_vel_pub.publish(self.twist)
            rospy.sleep(3.5)
            
            # Turn 90 degrees left
            self.stop_robot()
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.5
            self.cmd_vel_pub.publish(self.twist)
            rospy.sleep(3.2)
            
            self.stop_robot()
            self.state = State.RETURN_TO_LINE
        
        elif self.state == State.RETURN_TO_LINE:
            # Use OpenAI to find the line
            prompt = """
            Analyze this image and help the robot find the black line on the floor.
            Respond with EXACTLY ONE of these options:
            - MOVE_FORWARD (to explore)
            - TURN_LEFT
            - TURN_RIGHT
            - LINE_FOUND (black line is visible)
            - NO_LINE_VISIBLE
            
            IMPORTANT: Your response must be just one of these five options, nothing else.
            """
            
            response = self.call_openai_api(self.last_image, prompt)
            
            if response == "MOVE_FORWARD":
                self.twist.linear.x = 0.1
                self.twist.angular.z = 0.0
            elif response == "TURN_LEFT":
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.3
            elif response == "TURN_RIGHT":
                self.twist.linear.x = 0.0
                self.twist.angular.z = -0.3
            elif response == "LINE_FOUND":
                rospy.loginfo("Line found, resuming line following")
                self.state = State.FOLLOW_LINE
                self.object_type = ObjectType.UNKNOWN
            elif response == "NO_LINE_VISIBLE":
                # Continue searching with a default behavior
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.3
            
            self.cmd_vel_pub.publish(self.twist)
    
    def run(self):
        """Main run loop"""
        rospy.loginfo("AI Controller running...")
        
        while not rospy.is_shutdown():
            # Process handled in callbacks
            self.rate.sleep()

if __name__ == '__main__':
    try:
        controller = AIController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Clean up
        cv2.destroyAllWindows() 
