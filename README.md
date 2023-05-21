# An Exploration into PRM Maintenance in Dynamic 2D Environment
Studies how to maintain PRM algorithm in dynamic 2D (two-dimensional) environment, where obstacles 
occasionally get added to/deleted from the map.

This is a course project (_EE5058 - Introduction to Information Technology_) in SUSTech, which features on enhancing 
existing sampling-based motion planning methods, specifically PRM and RRT. The author is purely a rookie in the area
of Sampling-based Motion Planning. Therefore, it is suggested to carefully examine the content and code before using.

## Introduction

### Assumption

This project assumes the following statements:
1. The robot operates on a 2D rectangle map, and is not expected to leave the map range. To simplify, the edge of the
map is surrounded by 1-unit point obstacles.
2. This project utilizes _[PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)_ and considers the robot 
as well as the obstacles as rounds (represented as `(x, y, r)`).

### Problems to Examine

In detail, it examines the following problems:
1. How to maintain PRM when new obstacles are added to the map, blocking several existing edges.
2. How to maintain PRM when existing obstacles are deleted from the map, sparing extra free space for motion planning.
3. (Optional) How to build a road map (like the one in PRM) in an online fashion, in order to amortize the setup time.

## Dynamic PRM

### Collision Checking: BVH

Now that obstacles are rounds rather than points, a simple obstacle KD Tree does not give the actual nearest obstacle.
To avoid brute force collision checking, other data structures should be considered. Here, we choose BVH (Bounding
Volume Hierarchy).

### New Obstacles Blocking Existing Edges

xxx

### Existing Obstacles Deleted

xxx
