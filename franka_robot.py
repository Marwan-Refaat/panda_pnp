import string
import tf
import sys
import rospy
import actionlib
import numpy as np
import moveit_msgs.msg
import moveit_commander
from copy import deepcopy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from franka_gripper.msg import MoveAction
from franka_gripper.msg import HomingAction
from franka_gripper.msg import MoveActionGoal     # imported to populate goal with the correct syntax
from tf.transformations import quaternion_matrix, quaternion_from_matrix
import math

class Robot(object):

    #Init Method: Contains Subscribers & Publishers, Moveit Init, and Properties Definitions
    def __init__(self):
        super(Robot, self).__init__()

        #Topic Subscribers & Publishers
        self.posePub = rospy.Publisher('/target_pose',PoseStamped, queue_size=20) #Pose publisher to visualize pose
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

        #Init Moveit Related Classes
        self.robot_state = moveit_commander.RobotCommander()
        self.setup_planner()
        
        #Tf Listener
        self.listener = tf.TransformListener()

        #Properties Definitions
        #self.grippers = GripperClient()
        self.JOINT_BASE = 0
        self.JOINT_WRIST = 6
        self.ee_to_finger = 0.13
        self.replan = False
        self.marker_pose = None #Init marker pose as null

        #Robot Offsets
        self.hand_EE_offset=0.103      #Z distance from hand to EE frame
        #Offsets for x,y,z axes
        self.eps = {"x":0.0, 
                    "y":0.0,
                    "z":0.007
                    }
        #Gripper Offsets
        self.close_percentage = 0.65     #was 0.5
        self.grasp_clearence = 0.03    # was 0.02    
        self.finger_offset = 0.01

        #Saved Joint Targets
        self.saved_joint_targets = {"home":self.home_joints,"dropoff":self.dropoff_joints}
        self.dropoff_joints = [0.5504058568859961,0.721285769390586,0.738233269560398,-2.0060661936842368,-0.7433834850399285,2.3909292103977107,0.9083082512544473]
        self.home_joints = [0.05622759528301264,-0.5044851906676041,0.01182232086595736,-1.9919824140541247,-0.01685925059682793,1.5351335138195017,0.8131031781409933]

    #Moveit Planner Init: Includes parameters for Moveit Planner and Scene Planning
    def setup_planner(self):
        
        #Init MoveGroup
        self.moveGroup = moveit_commander.MoveGroupCommander("panda_arm") 
        
        #Moveit Planning Parameters
        self.moveGroup.set_end_effector_link("panda_hand")# planning wrt to panda_EE or link8
        #self.moveGroup.set_end_effector_link("panda_gripper_center") #Set EE to panda_gripper_center, courtesy of https://github.com/ros-planning/moveit/issues/1694
        self.moveGroup.set_max_velocity_scaling_factor(0.15)  # scaling down velocity
        self.moveGroup.set_max_acceleration_scaling_factor(0.05)  # scaling down acceleration
        self.moveGroup.allow_replanning(True)
        self.moveGroup.set_num_planning_attempts(10)
        self.moveGroup.set_goal_position_tolerance(0.0005)
        self.moveGroup.set_goal_orientation_tolerance(0.01)
        self.moveGroup.set_planning_time(5)
        self.moveGroup.set_planner_id("FMTkConfigDefault")
        #self.moveGroup.set_planning_frame("panda_EE")
        #print (self.moveGroup.get_planning_frame())
        self.__move_group._g.start_state_monitor(1.0)  # wait = 1.0
        
        #Reset Hand Offset if planning to gripper center
        #TODO Remove after Testing
        if (self.moveGroup.get_end_effector_link() == "panda_gripper_center"):
            self.hand_EE_offset = 0

        print ("============ Moveit Planner Initialized ============")
        #Outputting Basic Info (From Tutorial at https://github.com/ros-planning/moveit_tutorials/tree/melodic-devel/doc/move_group_python_interface/scripts/move_group_python_interface_tutorial.py)

        #Planning Frame
        planning_frame = self.moveGroup.get_planning_frame()
        print ("============ Planning frame: %s" % planning_frame)

        #Current End Effector Link
        eef_link = self.moveGroup.get_end_effector_link()
        print ("============ End effector link: %s" % eef_link)

        #Planning Groups
        group_names = self.robot_state.get_group_names()
        print ("============ Available Planning Groups:", self.robot_state.get_group_names())

    
    #Generic function that takes joint angle array and moves robot accordingly
    def go_joint_location(self,joint_goal,confirm=False):
        """ 
        Moves to joint angles

        Args:
            `joint_goal` - 7x1 vector of joint angles in radians

            `confirm` - bool, asks for confirmation before performing action (Default=True)

        Returns:
        Void
    """  

        #If confirm is set to true, wait for confirmation to move to target
        if confirm:
            print(f'******** Press Y to move to joint target... ********')
            x = input()

            if x != "y":
                print('===== Move Aborted =====')
                return 0
        
        #Move to joint goal
        print(f"\n \n============ Moving to joint target... ============")
        self.moveGroup.go(joint_goal, wait=True)

        self.moveGroup.stop()
    
    def move_to(self,target_pose,confirm=True):
        """ 
        Moves to pose target

        Args:
            `target_pose` - ROS Pose message for target

            `confirm` - bool, asks for confirmation before performing action (Default=True)

        Returns:
        Void
    """  
        
        self.posePub.publish(target_pose)     #Publish pose to visualize target

        #Set pose target and plan 
        target = self.moveGroup.set_pose_target(target_pose)
        self.plan = self.moveGroup.plan()
        
        #Preview planned trajectory
        self.display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        self.display_trajectory.trajectory_start = self.moveGroup.get_current_state()
        self.display_trajectory.trajectory.append(self.plan)
        self.display_trajectory_publisher.publish(self.display_trajectory)


        print("===== Trajectory Planning Complete =====")
        #If confirm is set to true, wait for confirmation to move to target
        if confirm:
            print(f'******** Execute Pose Target Move? (Y/N) ********')
            x = input()

            if x.lower() != "y":
                print('===== Move Aborted =====')
                self.moveGroup.stop()
                self.moveGroup.clear_pose_targets()
                return 0
        
        print('===== Executing Move.... =====')
        self.moveGroup.execute(self.plan,wait=True)

    def current_pose(self):
        """ 
        Gets current robot pose
        
        Args:

        Returns:
        Current pose of end effector
        """
        #Calling twice to fix issue: https://github.com/ros-planning/moveit/issues/2715
        current_pose = self.moveGroup.get_current_pose()  
        current_pose = self.moveGroup.get_current_pose()

        return current_pose
    
        
    def home_gripper(self):
        print ("============ Homing gripper... ============")
        client = actionlib.SimpleActionClient("/franka_gripper/homing", HomingAction)
        client.wait_for_server()
        client.send_goal(True)
        homing_done = client.wait_for_result()
        
        return homing_done

    def move_gripper_(self, w, s):
        client = actionlib.SimpleActionClient("/franka_gripper/move", MoveAction)
        client.wait_for_server()
        goal = MoveActionGoal
        goal.width = w
        goal.speed = s
        print ("============ Moving gripper to set width=", goal.width, "set speed=", goal.speed," ============")
        client.send_goal(goal)
        
        move_done = client.wait_for_result()

        return move_done

    def close_gripper(self, diameter):
        diameter = diameter - self.finger_offset + self.grasp_clearence
        self.move_gripper_(diameter, 0.02)  # width, speed 

    def open_gripper(self):
        diameter = 0.08
        diameter = diameter - self.finger_offset + self.grasp_clearence
        self.move_gripper_(diameter, 0.02)  # width, speed  
