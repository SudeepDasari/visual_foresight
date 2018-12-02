#!/bin/bash
while true
do
  rosrun intera_interface enable_robot.py -e
  rosrun intera_examples set_interaction_options.py -r 10 -k 0.05 0.05 1600 10 10 10 -m 1 1 0 1 1 1
  #rosrun intera_examples set_interaction_options.py -r 10 -k 1600 1600 800 10 10 10 -m 0 0 0 1 1 1
done


