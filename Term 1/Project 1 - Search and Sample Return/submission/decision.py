import numpy as np
from time import sleep


#
# Does the work for the 'explore' mode
# As long as there is a path ahead it will try to move on that path
# with a slight bias towards the left side; this is a simple way of
# avoiding going in circles and exploring the whole map
# if too litle navigation path is available it will go in 'stop' mode
# if a rock is detected it will stitch to 'collect' mode
#
def explore_mode(Rover):
	# check if there are any samples to pick and they are on the left side
	# if they are on the right side they will be picked up when we come back
	if len(Rover.samp_angles) > 3:
		Rover.throttle = 0
		Rover.brake = 1 	# slight brake
		# we need to store the positon before pickup otherwise there is a
		# very high risk that after pickup we will not be able to continue on
		# the same path
		Rover.save_yaw = Rover.yaw
		Rover.steer = np.clip(np.mean(Rover.samp_angles) * 180/np.pi, -15, 15)
		Rover.mode = 'collect'

	elif Rover.samples_collected > 5 and Rover.perc_mapped > 0.95:
		Rover.mode = 'return'
		
	# Check the extent of navigable terrain
	elif len(Rover.nav_angles) >= Rover.stop_forward:
		# If mode is forward, navigable terrain looks good
		# and velocity is below max, then throttle
		if Rover.vel < Rover.max_vel:
			Rover.throttle = Rover.throttle_set
		else: # Else coast
			Rover.throttle = 0
		Rover.brake = 0
		# Set steering to average angle clipped to the range +/- 15
		nav_mean = np.mean(Rover.nav_angles)
		nav_std = np.std(Rover.nav_angles)
		steer = nav_mean + nav_std / 2.25	 # slight bias to drive on the left side
		steer = steer * 0.6				   # acts like a P factor (PID); reduces overshooting
		Rover.steer = np.clip(steer * 180/np.pi, -15, 15)
	# If there's a lack of navigable terrain pixels then go to 'stop' mode
	elif len(Rover.nav_angles) < Rover.stop_forward:
			# Set mode to "stop" and hit the brakes!
			Rover.throttle = 0
			# Set brake to stored brake value
			Rover.brake = Rover.brake_set
			Rover.steer = 0
			Rover.mode = 'stop'
	# make sure we return the updated Rover
	return Rover


#
# deals with the 'stop' mode
# means we have an obstacle and we need to avoid it
#
def stop_mode(Rover):
	# If we're in stop mode but still moving keep braking
	if Rover.vel > 0.2:
		Rover.throttle = 0
		Rover.brake = Rover.brake_set
		Rover.steer = 0
	# If we're not moving (vel < 0.2) then do something else
	elif Rover.vel <= 0.2:
		# Now we're stopped and we have vision data to see if there's a path forward
		if len(Rover.nav_angles) < Rover.go_forward:
			Rover.throttle = 0
			Rover.brake = 0			# Release the brake to allow turning
			Rover.steer = -15	   # we trun to the left to keep aligned with
									# the fact that we drive on the left side
		# If we're stopped but see sufficient navigable terrain in front then go!
		if len(Rover.nav_angles) >= Rover.go_forward:
			Rover.throttle = Rover.throttle_set # Set throttle back to stored value
			Rover.brake = 0						# Release the brake
			# Set steer to mean angle
			Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
			Rover.mode = 'explore'
	# make sure we return the updated Rover
	return Rover


def collect_mode(Rover):
	# near sample?
	if Rover.near_sample > 0:
		Rover.brake = 5
		Rover.throttle = 0
		Rover.send_pickup = True
		Rover.mode = 'pick'

	# not yet; move closer
	else:
		Rover.brake = 0
		if Rover.vel < Rover.max_vel / 2.0:			# go slower
			Rover.throttle = Rover.throttle_set
		else: # Else coast
			Rover.throttle = 0
		if len(Rover.samp_angles) > 0:
			mean_dir = np.mean(Rover.samp_angles)
			steer_dir = mean_dir * 180 / np.pi
			Rover.steer = np.clip(steer_dir, -15, 15) 
	# make sure we return the updated Rover
	return Rover


#

def done_pick_mode(Rover):
	# we finished picking and we need to reorient the rover in the direction
	# we were initially
	Rover.throttle = 0
	Rover.brake = 0
	#print('save: %4.2f; actual: %4.2f; comm: %4.2f' % (Rover.save_yaw, Rover.yaw, Rover.save_yaw - Rover.yaw))
	if abs(Rover.save_yaw - Rover.yaw) > 2.0 :	
		Rover.steer = np.clip(Rover.save_yaw - Rover.yaw, -15, 15)
	else:
		Rover.steer = 0 
		Rover.mode = 'explore'
   # make sure we return the updated Rover
	return Rover

#
# returns to center
# just cruise until close enough to the map start
#
def return_mode(Rover):
	# calculates distance from start
	dx = Rover.pos[0] - Rover.return_pos[0]
	dy = Rover.pos[1] - Rover.return_pos[1]
	dist = np.sqrt(dx**2 + dy**2)
	if dist < 10:
		# we are close enough
		Rover.mode = 'finish'
		
	elif len(Rover.nav_angles) >= Rover.stop_forward:
		# navigate forward - will get them eventually
		if Rover.vel < Rover.max_vel:
			Rover.throttle = Rover.throttle_set
		else: # Else coast
			Rover.throttle = 0
		Rover.brake = 0
		# Set steering to average angle clipped to the range +/- 15
		nav_mean = np.mean(Rover.nav_angles)
		steer = nav_mean * 0.6				   # acts like a P factor (PID); reduces overshooting
		Rover.steer = np.clip(steer * 180/np.pi, -15, 15)
	# If there's a lack of navigable terrain pixels then go to 'stop' mode
	elif len(Rover.nav_angles) < Rover.stop_forward:
			# Set mode to "stop" and hit the brakes!
			Rover.throttle = 0
			# Set brake to stored brake value
			Rover.brake = Rover.brake_set
			Rover.steer = 0
			Rover.mode = 'stop'
	# make sure we return the updated Rover
	return Rover



# This is where you can build a decision tree for determining throttle, brake and steer
# commands based on the output of the perception_step() function
def decision_step(Rover):

	# Implement conditionals to decide what to do given perception data
	# Here you're all set up with some basic functionality but you'll need to
	# improve on this decision tree to do a good job of navigating autonomously!

	if Rover.nav_angles is None:
		Rover.throttle = Rover.throttle_set
		Rover.steer = 0
		Rover.brake = 0

	else:
		# Check for Rover.mode status
		if Rover.mode == 'start':
			# save start position so we can return
			Rover.return_pos = Rover.pos
			Rover.mode = 'explore'

		elif Rover.mode == 'explore':
			Rover = explore_mode(Rover)

		elif Rover.mode == 'stop':
			Rover = stop_mode(Rover)

		elif Rover.mode == 'collect':
			Rover = collect_mode(Rover)

		elif Rover.mode == 'pick':
			sleep(10)			 # we need this to deal with some dalays in operation
			Rover.mode = 'done_pick' # finished picking
			# otherwise just wait for the pickup to complete

		elif Rover.mode == 'done_pick':
			Rover = done_pick_mode(Rover)

		elif Rover.mode == 'return':
			Rover = return_mode(Rover)

		elif Rover.mode == 'finish':
			# we're done; stop the rover
			Rover.brake = 5
			Rover.thorottle = 0


	# If in a state where want to pickup a rock send pickup command
	#if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
	#	 Rover.send_pickup = True

	return Rover	# Implement conditionals to decide what to do given perception data
