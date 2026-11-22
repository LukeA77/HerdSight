# One still taken during a survey. The geotag is PX4's own position estimate at the trigger time
# (never the simulator's true pose), so home base sees what it would see on a real flight.
std_msgs/Header header             # stamp = trigger time
uint32 index                       # photo index in the plan
geometry_msgs/Point position_estimate   # local east-north-up metres, from PX4's estimator
float32 height_agl_m
float32 yaw_rad
float32 gimbal_pitch_rad
string image_path
string truth_path                  # simulation only: visible animals from the bounding-box camera
