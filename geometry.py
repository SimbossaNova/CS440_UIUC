# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy


def orientation(p, q, r):
    """Helper function to determine orientation of triplet (p, q, r).
       Returns 0 if collinear, 1 if clockwise, 2 if counterclockwise.
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: 
        return 0
    return 1 if val > 0 else 2

def on_segment(p, q, r):
    """Helper function to check if point q lies on line segment pr."""
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    if alien.is_circle():
        center = alien.get_centroid()
        radius = alien.get_width()
        
        for wall in walls:
            # Compute distance from the center of the circle to the wall segment
            distance = point_segment_distance(center, ((wall[0], wall[1]), (wall[2], wall[3])))
            
            # If the distance is less than the radius, the alien touches the wall
            if distance < radius:
                return True
    else:
        # If the alien is in oblong (sausage) form
        head, tail = alien.get_head_and_tail()
        alien_segment = (head, tail)
        width = alien.get_width()
        
        for wall in walls:
            wall_segment = ((wall[0], wall[1]), (wall[2], wall[3]))
            
            # Check if the alien's main body segment intersects with the wall
            if do_segments_intersect(alien_segment, wall_segment):
                return True
            
            # Check distances to endpoints (head and tail) of the alien segment from the wall
            if point_segment_distance(head, wall_segment) < width or point_segment_distance(tail, wall_segment) < width:
                return True
                
    # If none of the conditions are met, the alien doesn't touch the wall
    return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    # Unpack window dimensions
   width, height = window
   
   # If the alien is in circle form
   if alien.is_circle():
       center = alien.get_centroid()
       radius = alien.get_width()
       
       # Check if the circle is out of the window's boundaries
       if center[0] - radius < 0 or center[0] + radius > width or center[1] - radius < 0 or center[1] + radius > height:
           return False
   else:
       # If the alien is in oblong (sausage) form
       head, tail = alien.get_head_and_tail()
       alien_width = alien.get_width()
       
       # Check if any of the alien's endpoints or its width is out of the window's boundaries
       if (head[0] - alien_width < 0 or head[0] + alien_width > width or
           tail[0] - alien_width < 0 or tail[0] + alien_width > width or
           head[1] - alien_width < 0 or head[1] + alien_width > height or
           tail[1] - alien_width < 0 or tail[1] + alien_width > height):
           return False
       

       
   # If none of the conditions are met, the alien is within the window
   return True



def is_point_in_polygon(point, polygon):
    
    n = len(polygon)
    if n < 3:
        return False
    extreme = (1e10, point[1])
    count = 0
    i = 0
    while True:
        next_point = (i + 1) % n
        segment1 = ((point[0], point[1]), extreme)
        segment2 = ((polygon[i][0], polygon[i][1]), (polygon[next_point][0], polygon[next_point][1]))
        
        # Debug print
        print(f"Segment 1: {segment1}, Segment 2: {segment2}")
        
        if do_segments_intersect(segment1, segment2):
            if orientation(polygon[i], point, polygon[next_point]) == 0:
                return on_segment(polygon[i], point, polygon[next_point])
            count += 1
        i = next_point
        if i == 0:
            break
    return count % 2 == 1

def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    path = (alien.get_centroid(), waypoint)
    for wall in walls:
       if do_segments_intersect(path, ((wall[0], wall[1]), (wall[2], wall[3]))):
           return True
    return False

def point_segment_distance(p, s):
    # Unpack point and segment
    px, py = p
    (x1, y1), (x2, y2) = s
    
    # Compute vector from segment's start to the point
    dx, dy = px - x1, py - y1
    
    # Compute the squared length of the segment
    segment_length_squared = (x2 - x1)**2 + (y2 - y1)**2
    
    # If the segment's length is zero (i.e., it is a point), return the distance between the point and segment start
    if segment_length_squared == 0:
        return np.sqrt(dx**2 + dy**2)
    
    # Compute the t parameter for the projection (t = [(P-A) . (B-A)] / |B-A|^2)
    t = max(0, min(1, (dx * (x2 - x1) + dy * (y2 - y1)) / segment_length_squared))
    
    # Compute the projection's (x, y) coordinates
    projection_x = x1 + t * (x2 - x1)
    projection_y = y1 + t * (y2 - y1)
    
    # Return the distance between the point and the projection
    return np.sqrt((px - projection_x)**2 + (py - projection_y)**2)


def do_segments_intersect(s1, s2):
   def orientation(p, q, r):
        """Find orientation of triplet (p, q, r).
        Returns:
            0: if p, q, and r are colinear
            1: if they are in clockwise orientation
            2: if they are in counterclockwise orientation
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0: return 0
        return 1 if val > 0 else 2
    
   def on_segment(p, q, r):
        """Check if point q lies on segment pr."""
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
    
    # Unpack the segments
   p1, q1 = s1
   p2, q2 = s2
    
    # Compute orientations
   o1 = orientation(p1, q1, p2)
   o2 = orientation(p1, q1, q2)
   o3 = orientation(p2, q2, p1)
   o4 = orientation(p2, q2, q1)
    
    # General case of intersection
   if o1 != o2 and o3 != o4:
        return True
    
    # Special cases of intersection
    # p1, q1, and p2 are colinear and p2 lies on segment p1q1
   if o1 == 0 and on_segment(p1, p2, q1): return True
    # p1, q1, and q2 are colinear and q2 lies on segment p1q1
   if o2 == 0 and on_segment(p1, q2, q1): return True
    # p2, q2, and p1 are colinear and p1 lies on segment p2q2
   if o3 == 0 and on_segment(p2, p1, q2): return True
    # p2, q2, and q1 are colinear and q1 lies on segment p2q2    if o4 == 0 and on_segment(p2, q1, q2): return True
    
   return False


def segment_distance(s1: Tuple[Tuple[float, float], Tuple[float, float]], 
                     s2: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
    """
    Compute the distance from segment1 to segment2.  
    
    Args:
        s1: A tuple of coordinates indicating the endpoints of segment1.
        s2: A tuple of coordinates indicating the endpoints of segment2.

    Return:
        Euclidean distance between the two line segments.
    """
    
    # If segments intersect, distance is 0
    if do_segments_intersect(s1, s2):
        return 0.0
    
    # Compute all pairwise distances between endpoints of the two segments
    distances = [
        point_segment_distance(s1[0], s2),
        point_segment_distance(s1[1], s2),
        point_segment_distance(s2[0], s1),
        point_segment_distance(s2[1], s1)
    ]
    
    # Return the minimum distance
    return min(distances)


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
