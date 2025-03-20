# Cleaning Robot

### Task Breakdown:

---

### **1. Initialize the System:**

- **Set up ROS Node**: Start the ROS node for the robot.
- **Create Subscribers and Publishers**:
  - Subscribe to the camera feed (`/camera/color/image_raw`) to receive images from the robot's camera.
  - Publish velocity commands (`/cmd_vel`) to control the robot's movement.

---

### **2. Start Line Following:**

#### **Image Processing:**

- Capture images using the camera and process them to follow the line.
- Crop the lower half of the image to focus on the floor (where the line is).
- Convert the image to **HSV (Hue, Saturation, Value)** for better color segmentation.
- Apply a threshold to identify the **white line** used for line following.

#### **Calculate the Error:**

- Find the center of the white region of the line.
- Calculate the error between the center of mass of the line and the center of the image.

#### **Adjust Robot’s Movement:**

- Based on the error, adjust the robot’s **linear** and **angular** velocities to keep the robot on the line.

---

### **3. Detect Pegs (Red, Green, Blue):**

#### **Image Processing for Peg Detection:**

- Convert the cropped image to **HSV**.
- Apply color masks to detect **red**, **green**, and **blue** pegs by defining their respective HSV color ranges.
- Find contours for each color (red, green, and blue) in the mask.

#### **Identify Pegs:**

- For each color, identify the largest contour (representing a peg).
- If a peg is detected and the robot is not currently carrying a peg, set the `has_peg` flag to **True** and simulate grabbing the peg.

---

### **4. Move Pegs to Correct Zone:**

#### **Determine Direction:**

- Check the current direction of the robot. The direction will determine whether the deploy zone is to the left or right of the robot.

#### **Move to Deploy Zone (for Green Peg):**

- If a green peg is detected, and the robot is carrying the peg (`has_peg = True`), decide which direction to go based on the robot’s position.
- Turn the robot to the correct direction (left or right).
- Move the robot to the deploy zone.

#### **Release the Green Peg:**

- Once in the deploy zone, stop the robot.
- Simulate releasing the green peg by setting `has_peg` to **False** and printing a message to indicate the release.

#### **Move Red or Blue Pegs Outside the Perimeter:**

- If a red or blue peg is detected, and the robot is carrying the peg, turn the robot in the opposite direction (left or right) to move outside the perimeter.
- Move the robot outside the perimeter (simulated by moving for a short duration).

#### **Release Red or Blue Peg:**

- Once outside the perimeter, stop the robot and release the peg by setting `has_peg` to **False**.

---

### **5. Return to the Original Position:**

#### **Determine Position:**

- Track the original position before the robot leaves the line (when the peg is grabbed).

#### **Navigate Back to the Original Position:**

- Use the **line-following algorithm** to return to the original position after releasing the peg (whether it’s a green peg in the deploy zone or a red/blue peg outside the perimeter).

#### **Resume Line Following:**

- Once the robot returns to the original position, it resumes its line-following behavior.

---

### **Workflow Summary:**

- **Initialize** the system (ROS node, camera feed, and publishers).
- **Follow the line** using image processing and error correction.
- **Detect pegs** (red, green, and blue) and grab them when detected.
- **Move the pegs** to the correct zone (green to the deploy zone, red/blue outside the perimeter) and release the peg and back away as to not tip it over when turning in text step.
- **Return** to the original position and **resume line following** after releasing the peg.

---
