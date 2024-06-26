import cv2
import numpy as np
import heapq

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

def detect_points(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper thresholds for green and red colors
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Threshold the image to get green and red regions
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    red_mask = cv2.inRange(hsv, lower_red2, upper_red2)

    # Find contours in the green and red masks
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the center points of green and red contours
    green_points = []
    red_points = []
    for contour in green_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            green_points.append((cX, cY))

    for contour in red_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            red_points.append((cX, cY))

    return green_points, red_points

def bitstar_search(image, start, end):
    # Create the start and end nodes
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize the open and closed lists for the forward search
    open_list = []
    closed_list = []

    # Add the start node to the open list
    heapq.heappush(open_list, start_node)

    # Create a copy of the original image
    img_copy = image.copy()

    # Set a maximum iteration limit
    max_iterations = 10000
    iterations = 0

    # Initialize the obstacle positions
    obstacle_positions = set()

    # Run the BiT* algorithm
    while open_list and iterations < max_iterations:
        iterations += 1

        # Get the current node from the open list
        current_node = heapq.heappop(open_list)

        # Add the current node to the closed list
        closed_list.append(current_node)

        # Check if the current node is the goal node
        if current_node == end_node:
            # Path found, trace back the path
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            path.reverse()
            return path

        # Generate the neighboring nodes
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Check if the new position is within the image boundaries
            if node_position[0] < 0 or node_position[0] >= image.shape[0] or node_position[1] < 0 or node_position[1] >= image.shape[1]:
                continue

            # Check the pixel color of the new position
            pixel_color = image[node_position[0], node_position[1]]

            # Check if the pixel color is black (obstacle)
            if np.array_equal(pixel_color, [0, 0, 0]):
                continue

            # Check if the new position collides with any obstacle
            if node_position in obstacle_positions:
                continue

            # Create a new node
            new_node = Node(current_node, node_position)

            # Calculate the cost to move to the neighbor
            new_node.g = current_node.g + 1
            new_node.h = ((new_node.position[0] - end_node.position[0]) ** 2) + (
                    (new_node.position[1] - end_node.position[1]) ** 2)
            new_node.f = new_node.g + new_node.h

            # Check if the neighbor is already in the open list
            existing_node = next((node for node in open_list if node == new_node), None)
            if existing_node is not None:
                if new_node.g < existing_node.g:
                    open_list.remove(existing_node)
                else:
                    continue

            # Check if the neighbor is already in the closed list
            existing_node = next((node for node in closed_list if node == new_node), None)
            if existing_node is not None:
                if new_node.g < existing_node.g:
                    closed_list.remove(existing_node)
                else:
                    continue

            # Add the neighbor to the open list
            heapq.heappush(open_list, new_node)

            # Add the neighbor to the obstacle positions
            obstacle_positions.add(new_node.position)

    return []  # No path found



# Load the image
image_path = '/content/Screenshot 2024-06-26 191545.png'
image = cv2.imread(image_path)

# Detect green and red points
green_points, red_points = detect_points(image)

# Select the first detected green and red points as start and end, respectively
start = green_points[0]
end = red_points[0]

# Apply the BiT* algorithm
path = bitstar_search(image, start, end)

# Print the path
print("Path:", path)

# Draw the path on the image
for position in path:
    cv2.circle(image, position, 2, (0, 255, 0), -1)

# Display the final image with the path
cv2_imshow(image)
cv2.waitKey(0)
cv2.destroyAllWindows()