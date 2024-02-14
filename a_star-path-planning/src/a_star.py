#!/usr/lib/bin/env/ python

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import rospy
import numpy as np
from gazebo_msgs.msg import LinkStates
import rostopic
import time


# every grid cell is a node object
class Node:
    def __init__(self, i, j, parent):
        self.i = i
        self.j = j
        self.parent = parent
        self.g_cost = 0
        self.h_cost = 0
        self.f_cost = 0


# 0 represents empty cells, traversable
# 1 represents obstacle cells
OCCUPANCY_GRID = np.rot90([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1],
    [1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1],
    [1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1],
    [1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
], 3)


LINK_STATES_TOPIC = "/gazebo/link_states"
HUSKY_LINK = "husky::base_link"
GAZEBO_NODE = "/gazebo"


def get_distance(neighbor, goal_node):
    # heuristic for the estimated cost between the current node and the goal
    # ϵ = 1
    # ϵ is the coefficient of heuristic cost
    # You can start with ϵ = 1, and
    # tune only if required, based on the paths returned by your A* algorithm.

    manhattan_distance = abs(neighbor.i - goal_node.i) + abs(neighbor.j - goal_node.j)
    # euclidean_distance = np.sqrt((neighbor.i - goal_node.i)**2 + (neighbor.j - goal_node.j)**2)

    return manhattan_distance


def callback(msg):
    print("METHOD: CALLBACK")

    # publishes a set of arrays containing the global position of all objects in the world
    # /husky/base link
    # husky’s position corresponds to the index in the Pose array
    # print(msg.name)
    # print(type(msg.name)) = list
    # print(msg.name.index(HUSKY_LINK)) = 36

    husky_link_index = msg.name.index(HUSKY_LINK)
    start_pose = msg.pose[husky_link_index]
    X = start_pose.position.x
    Y = start_pose.position.y
    # print(start_pose)

    start_i, start_j = get_grid_indices(X, Y)
    goal_i, goal_j = get_grid_indices(goal_x, goal_y)

    print("====================================================")
    print(start_i, start_j)
    print(goal_i, goal_j)
    print("====================================================")

    # call a star here
    path = astar(start_i, start_j, goal_i, goal_j)

    # plot path on matplotlib
    plot_on_matplotlib(path)

    subscriber.unregister()

    return


# the bounds of (i, j) are [0, 41], integers only
# the bounds of (X, Y) are [-10, 10], real numbers

def get_world_coordinates(i, j):
    # A function that takes (i, j) index of the map node and returns (X, Y) world coordinates of the center of the cell
    X = i / 2 - 10
    Y = j / 2 - 10
    return X, Y


def get_grid_indices(X, Y):
    # A function that takes (X, Y) world coordinates and returns (i, j) node indexes of the cell which (X, y) point falls in.
    i = int((X + 10) * 2)
    j = int((Y + 10) * 2)
    return i, j


# The global path essentially contains a list of nodes/checkpoints, such that
# when the robot moves to each one of them in succession, it will eventually reach the final
# goal location


def astar(start_i, start_j, goal_i, goal_j):
    print("METHOD: A*")

    # implement astar here

    """
    OPEN list   # the set of nodes to be evaluated
    CLOSED list     # the set of nodes already evaluated
    add the start node to OPEN

    loop
        current = node in OPEN with the lowest f_cost
        remove current from OPEN
        add current to CLOSED

        if current is the target node   # path has been found
            return

        foreach neighbour of the current node
            if neighbour is not traversable or neighbour is in closed
                skip to the next neighbour

            if new path to neighbour is shorter OR neighbour is not in OPEN
                set f_cost of neighbour     # by calculating the g_cost and h_cost
                set parent of neighbour to current
                if neighbour is not in OPEN
                    add neighbour to OPEN

    """

    # convert position into node
    start_node = Node(start_i, start_j, None)
    goal_node = Node(goal_i, goal_j, None)
    start_node.g_cost = 0
    start_node.h_cost = 0
    start_node.f_cost = 0
    goal_node.g_cost = 0
    goal_node.h_cost = 0
    goal_node.f_cost = 0

    # open and closed sets
    open = []
    closed = []
    open.append(start_node)

    while len(open) > 0:

        # current node is the one with lowest f_cost or h_cost if f_cost is same
        current_node = open[0]
        for open_node in open:
            if open_node.f_cost < current_node.f_cost or (
                open_node.f_cost == current_node.f_cost
                and open_node.h_cost < current_node.h_cost
            ):
                current_node = open_node

        open.remove(current_node)
        closed.append(current_node)

        # reached the goal
        if are_two_nodes_equal(current_node, goal_node):
            return get_path(current_node)

        # get 8 neighbors
        neighbors = []
        for adjacent_cell in [
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            neighbor_pos = (
                current_node.i + adjacent_cell[0],
                current_node.j + adjacent_cell[1],
            )

            # neighbor should be the accessible cells inside the grid
            if (
                OCCUPANCY_GRID[neighbor_pos[0]][neighbor_pos[1]] != 0
                or neighbor_pos[0] > 41
                or neighbor_pos[0] < 0
                or neighbor_pos[1] > 41
                or neighbor_pos[1] < 0
            ):
                continue

            new_node = Node(neighbor_pos[0], neighbor_pos[1], current_node)
            neighbors.append(new_node)

        # [print(node.i, node.j) for node in neighbors]
        # break

        # iterate thorugh neighbors
        for neighbor in neighbors:

            # neighbor is on the closed list
            for closed_child in closed:
                if are_two_nodes_equal(neighbor, closed_child):
                    continue

            neighbor.g_cost = (
                current_node.g_cost + 1
            )  # distance between current and start node
            neighbor.h_cost = get_distance(
                neighbor, goal_node
            )  # heuristic cost from goal
            neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
            # print(neighbor.f_cost, current_node.i, current_node.j)

            # neighbor is already in the open list
            for open_node in open:
                if (
                    are_two_nodes_equal(neighbor, open_node)
                    and neighbor.g_cost > open_node.g_cost
                ):
                    continue

            open.append(neighbor)

    return


def are_two_nodes_equal(node1, node2):
    return node1.i == node2.i and node1.j == node2.j


def get_path(goal):
    path = []
    while goal is not None:
        path.append((goal.i, goal.j))
        goal = goal.parent
    path.reverse()
    return path


def retrace_path(start_node, end_node):
    path = []
    current_node = end_node

    while current_node != start_node:
        path.append(current_node)
        current_node = current_node.parent

    path.reverse()
    return path


def plot_on_matplotlib(path):
    # Once a path is generated, plot it using matplotlib.

    for cell in path:
        OCCUPANCY_GRID[cell[0]][cell[1]] = -1

    plt.figure(figsize=(12, 12))
    plt.xticks(np.arange(-0.5, 41.5, 1))
    plt.yticks(np.arange(-0.5, 41.5, 1))
    plt.tick_params(
        axis="x",
        which="major",
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False,
    )
    plt.tick_params(
        axis="y",
        which="major",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )
    plt.grid(which="major", linewidth="0.4")
    plt.imshow(OCCUPANCY_GRID)
    plt.show()
    return


def move_to_next_cell():
    # first change direction to next cell
    # then move forward
    return


def subscribe():
    print("METHOD: SUBSCRIBE")

    # subscribe to /gazebo/link states to get global location
    global subscriber
    subscriber = rospy.Subscriber(
        name=LINK_STATES_TOPIC, data_class=LinkStates, callback=callback
    )
    rospy.spin()
    return


def is_topic_available():
    topics_list = rostopic.get_topic_list()[0]
    topics = list(map(lambda x: x[0], topics_list))
    # print("==============================================")
    # print(topics)
    # print("==============================================")
    return LINK_STATES_TOPIC in topics


if __name__ == "__main__":
    try:
        while True:
            if is_topic_available():
                break
            time.sleep(1)

        rospy.init_node("a_star", anonymous=True)

        global goal_x, goal_y
        goal_x = rospy.get_param("~goalx")
        goal_y = rospy.get_param("~goaly")

        # driver method
        subscribe()
    except rospy.ROSInterruptException:
        pass
