import cv2
import numpy as np
import heapq

green_points = None
red_points = None
blue_points = None
def find_colored_points(image, lower_color, upper_color):
    mask = cv2.inRange(image, lower_color, upper_color)
    coordinates = cv2.findNonZero(mask)
    return coordinates

def is_valid(x, y, rows, cols, obstacles):
    return 0 <= x < rows and 0 <= y < cols and not obstacles[x, y]

def dijkstra(image, start, end, obstacles):
    rows, cols = image.shape[:2]
    dist = np.full((rows, cols), np.inf)
    dist[start] = 0

    priority_queue = [(0, start)]
    heapq.heapify(priority_queue)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while priority_queue:
        current_dist, (x, y) = heapq.heappop(priority_queue)

        if (x, y) == end:
            break

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny, rows, cols, obstacles):
                new_dist = current_dist + 1
                if new_dist < dist[nx, ny]:
                    dist[nx, ny] = new_dist
                    heapq.heappush(priority_queue, (new_dist, (nx, ny)))

    return dist

def reconstruct_path(dist, start, end):
    path = []
    current = end
    while current != start:
        path.append(current)
        x, y = current
        min_dist = dist[x, y]
        next_step = None
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < dist.shape[0] and 0 <= ny < dist.shape[1] and dist[nx, ny] < min_dist:
                min_dist = dist[nx, ny]
                next_step = (nx, ny)
        current = next_step
    path.append(start)
    return path[::-1]

def main():
    # Read the image
    image_path = '/content/Screenshot 2024-06-26 191545.png'  # Replace with your image path
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Image not found or unable to open.")
        return

    rows, cols = image.shape[:2]

    # Define color ranges
    lower_green = np.array([0, 150, 0])
    upper_green = np.array([100, 255, 100])
    lower_red = np.array([0, 0, 150])
    upper_red = np.array([100, 100, 255])
    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([250, 250, 250])

    # Find green starting point
    green_points = find_colored_points(image, lower_green, upper_green)
    if green_points is not None:
        start = tuple(green_points[0][0][::-1])
        print(f"Green starting point: {start}")
    else:
        print("Green starting point not found.")
        return

    # Find red endpoint
    red_points = find_colored_points(image, lower_red, upper_red)
    if red_points is not None:
        end = tuple(red_points[0][0][::-1])
        print(f"Red endpoint: {end}")
    else:
        print("Red endpoint not found.")
        return

    # Identify blue obstacles
    obstacles = np.zeros((rows, cols), dtype=bool)
    blue_points = find_colored_points(image, lower_blue, upper_blue)
    if blue_points is not None:
        for point in blue_points:
            x, y = point[0][::-1]
            if 0 <= x < cols and 0 <= y < rows:
                obstacles[y, x] = True
        print(f"Number of blue obstacle points: {len(blue_points)}")
    else:
        print("No blue obstacle points found.")

    # Apply Dijkstra's algorithm
    dist = dijkstra(image, start, end, obstacles)
    
    if np.isinf(dist[end]):
        print("No path found.")
    else:
        print(f"Shortest distance: {dist[end]}")
        path = reconstruct_path(dist, start, end)
        print(f"Path: {path}")

        # Visualize the path
        for (x, y) in path:
            image[x, y] = [0, 255, 0]  # Yellow path

        cv2_imshow(image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
