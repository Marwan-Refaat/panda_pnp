import copy
import os
import time

import numpy as np
import rospy
from geometry_msgs.msg import Pose

from franka_robot import Robot
from utils.transforms import rotate_pose_msg_by_euler_angles, get_pose


class PickAndPlace:
    def __init__(
            self,
            place_position,
            force_threshold=-8,
            hover_distance=0.12,
            step_size=0.05,
    ):
        """

        @param place_position: Place position as [x, y, z]
        @param force_threshold: Z force threshold in Newtons
        @param hover_distance: Distance above the pose in meters
        @param step_size: Step size for approaching the pose
        """
        self.place_position = place_position
        self.force_threshold = force_threshold
        self._hover_distance = hover_distance
        self.step_size = step_size

        self.robot = Robot()

        homedir = os.path.join(os.path.expanduser('~'), "grasp-comms")
        self.grasp_request = os.path.join(homedir, "grasp_request.npy")
        self.grasp_available = os.path.join(homedir, "grasp_available.npy")
        self.grasp_pose = os.path.join(homedir, "grasp_pose.npy")

    def _approach(self, pose):
        """
        Move to a pose with a hover-distance above the requested pose and
        then move to the pose incrementally while monitoring the z force
        """
        print('approaching...')
        approach = copy.deepcopy(pose)
        approach.position.z = approach.position.z + self._hover_distance + self.robot.eps.z + self.robot.hand_EE_offset
        self.robot.move_to(approach)

        while approach.position.z >= pose.position.z:
            approach.position.z = approach.position.z - self.step_size
            self.robot.move_to(approach)


    def _retract(self):
        """
        Retract up from current pose
        """
        # Retrieve current pose from endpoint
        current_pose = self.robot.current_pose()

        #Add hover distance
        current_pose.position.z += self._hover_distance

        #Servo up from current pose
        self.robot.move_to(current_pose)

    def pick(self, grasp_pose):
        """
        Pick from given pose
        """
        # Calculate grasp pose
        pose = get_pose(position=grasp_pose[:3])

        # Apply grasp angle from model output
        pose = rotate_pose_msg_by_euler_angles(pose, 0.0, 0.0, grasp_pose[3])

        #Open gripper
        self.robot.open_gripper()

        #Approach pose
        self._approach(pose)

        #Close Gripper
        self.robot.close_gripper()

        #Retract from object
        self._retract()

    def place(self, place_position):
        """
        Place to given pose
        """
        # Calculate pose from place position
        pose = get_pose(position=place_position)

        # approach to the pose
        self._approach(pose)
        # open the gripper
        self.robot.open_gripper()
        # Get the next grasp pose
        np.save(self.grasp_request, 1)
        # retract to clear object
        self._retract()

    def run(self):
        #TODO Verify Connection with Robot is good
        #self.robot.connect()

        #Calibrate gripper
        self.robot.home_gripper()

        # Initialize grasp request and grasp available
        np.save(self.grasp_request, 0)
        np.save(self.grasp_available, 0)

        # Move robot to home pose
        print('Moving to start position...')
        self.robot.go_joint_location(self.robot.saved_joint_targets["home"])
        self.robot.open_gripper()

        # Get the first grasp pose
        np.save(self.grasp_request, 1)

        while not rospy.is_shutdown():
            print('Waiting for grasp pose...')
            while not np.load(self.grasp_available) and not rospy.is_shutdown():
                time.sleep(0.1)
            grasp_pose = np.load(self.grasp_pose)
            np.save(self.grasp_available, 0)

            # Perform pick
            print('Picking from ', grasp_pose)
            self.pick(grasp_pose)

            # Perform place
            print('Placing to ', self.place_position)
            self.place(self.place_position)
