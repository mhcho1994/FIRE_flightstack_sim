#!/usr/bin/env python3

__author__ = "Minhyun Cho"
__contact__ = "@purdue.edu"

# python packages and modules
import time 
import subprocess 
import os 
import time
import argparse

# check whether px4 is already built 
def check_build(directory_path): 
	if not os.path.exists(directory_path+'/build'):    
		output = subprocess.run(['make', 'px4_sitl'], capture_output=True, text=True, cwd=directory_path) 
		print(output.stdout)
	else:
		print('px4: already built')

# open terminator 
def open_terminator(command):
	subprocess.Popen(['terminator', '--new-tab', '-e' , command])
	

# main function
def main():

    # argument parser for setting up the mode
	parser = argparse.ArgumentParser(description='delivering running mode information')
	parser.add_argument('--mode','-m',type=str,default='sitl',help='sitl/mxexp')
	argin = parser.parse_args()

	# px4 build check 
	file_dir 	= 	os.path.dirname(os.path.realpath('__file__'))
	parent_dir 	=	os.path.abspath(os.path.join(file_dir, os.pardir))
	px4_directory = os.path.join(parent_dir,'work/ros2_ws/px4')
	check_build(px4_directory)

	# Open Terminator shells and run the commands 
	# NOTE: Gazebo coordinate is ENU
	print('run SiTL')
	if getattr(argin,'mode') == 'sitl':
		run_commands = [
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0.0,5.0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 1',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="5.0,5.0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 2',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="5.0,-5.0" PX4_GZ_MODEL=x500r ./build/px4_sitl_default/bin/px4 -i 3',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0.0,-5.0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 4',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="-5.0,-5.0" PX4_GZ_MODEL=x500r ./build/px4_sitl_default/bin/px4 -i 5',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="-5.0,5.0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 6',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0.0,0.0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 7',
			'MicroXRCEAgent udp4 -p 8888']
		
	elif getattr(argin,'mode') == 'mxsitl':
		run_commands = [
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="3.46,2.0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 1',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0.0,-4,0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 2',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="-3.46,2.0" PX4_GZ_MODEL=x500r ./build/px4_sitl_default/bin/px4 -i 3',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0.0,4.0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 4',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="3.46,-2.0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 5',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="-3.46,-2.0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 6',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0.0,0.0" PX4_GZ_MODEL=x500r ./build/px4_sitl_default/bin/px4 -i 7',
			'MicroXRCEAgent udp4 -p 8888']
		
	elif getattr(argin,'mode') == 'mxexp':
		run_commands = [
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="3.46,2.0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 1',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0.0,-4,0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 2',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="-3.46,2.0" PX4_GZ_MODEL=x500r ./build/px4_sitl_default/bin/px4 -i 3',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0.0,4.0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 4',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="3.46,-2.0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 5',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="-3.46,-2.0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 6',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0.0,0.5" PX4_GZ_MODEL=x500r ./build/px4_sitl_default/bin/px4 -i 7',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0.0,-0.5" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 8',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0.0,-1.0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 9',
			'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0.0,0.0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 10',
			'MicroXRCEAgent udp4 -p 8888']
		
	for idx, command in enumerate(run_commands): 

		if idx==0: 
			open_terminator(command) 
			time.sleep(6.0) 
		else: 
			open_terminator(command)
			time.sleep(3.0) 


# run main function
if __name__ == '__main__':

	main()

	# run_commands = [
	# 	'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="2.6,1.5" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 1',
	# 	'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0.0,-3,0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 2',
	# 	'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="-2.6,1.5" PX4_GZ_MODEL=x500r ./build/px4_sitl_default/bin/px4 -i 3',
	# 	'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0.0,3.0" PX4_GZ_MODEL=x500r ./build/px4_sitl_default/bin/px4 -i 4',
	# 	'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="2.6,-1.5" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 5',
	# 	'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="-2.6,-1.5" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 6',
	# 	'source ros2_ws/install/setup.bash; cd ros2_ws/px4; PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0.0,0.0" PX4_GZ_MODEL=x500g ./build/px4_sitl_default/bin/px4 -i 7',
	# 	'MicroXRCEAgent udp4 -p 8888']
