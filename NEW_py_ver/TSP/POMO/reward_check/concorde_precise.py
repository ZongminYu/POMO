import os
import math

def read_tsp_file(file_path):
    """
    从 .tsp 文件中读取节点坐标。
    :param file_path: .tsp 文件的路径
    :return: 一个字典，键是节点编号，值是 (x, y) 坐标
    """
    coordinates = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        start_reading = False
        for line in lines:
            if line.strip() == "NODE_COORD_SECTION":
                start_reading = True
                continue
            if start_reading:
                parts = line.strip().split()
                if len(parts) == 3:
                    node_id, x, y = parts
                    coordinates[int(node_id)] = (float(x), float(y))
    return coordinates

def read_sol_file(file_path):
    """
    从 .sol 文件中读取节点序列。
    :param file_path: .sol 文件的路径
    :return: 一个列表，包含节点序列
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # 跳过第一行（节点个数）
        tour = [int(num) for line in lines[1:] for num in line.strip().split()]
    return tour

def calculate_tsp_distance(coordinates, tour):
    """
    计算 TSP 距离。
    :param coordinates: 节点坐标字典
    :param tour: 节点序列
    :return: TSP 距离
    """
    total_distance = 0.0
    for i in range(len(tour)):
        node1 = tour[i] + 1
        node2 = tour[(i + 1) % len(tour)] + 1
        x1, y1 = coordinates[node1]
        x2, y2 = coordinates[node2]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # print("node1:", node1, "coordinates:", (x1, y1) , "node2:", node2, "coordinates:", (x2, y2), "distance:", distance)
        total_distance += distance
    return total_distance

def main():
    problems_dir = "/opt/yzm_xp/TSP_precise/database/random01/problems"
    solutions_dir = "/opt/yzm_xp/TSP_precise/database/random01/solutions"
    
    # 获取所有问题文件
    tsp_files = [f for f in os.listdir(problems_dir) if f.endswith(".tsp")]
    
        # 每1000个问题计算一次局部平均值
    chunk_size = 1000
    total_distances = []
    num_chunks = len(tsp_files) // chunk_size
    remaining_files = len(tsp_files) % chunk_size

    for i in range(num_chunks):
        chunk_distances = []
        for tsp_file in tsp_files[i * chunk_size: (i + 1) * chunk_size]:
            problem_path = os.path.join(problems_dir, tsp_file)
            solution_path = os.path.join(solutions_dir, tsp_file.replace(".tsp", ".sol"))
            
            # 读取问题和解决方案
            coordinates = read_tsp_file(problem_path)
            tour = read_sol_file(solution_path)
            
            # 计算 TSP 距离
            distance = calculate_tsp_distance(coordinates, tour)
            chunk_distances.append(distance)
        
        # 计算当前块的平均距离
        chunk_mean = sum(chunk_distances) / len(chunk_distances)
        total_distances.append(chunk_mean)
        print(f"Chunk {i + 1}/{num_chunks}, Mean Distance: {chunk_mean:.6f}")
    
    # 处理剩余的文件
    if remaining_files > 0:
        chunk_distances = []
        for tsp_file in tsp_files[num_chunks * chunk_size:]:
            problem_path = os.path.join(problems_dir, tsp_file)
            solution_path = os.path.join(solutions_dir, tsp_file.replace(".tsp", ".sol"))
            
            # 读取问题和解决方案
            coordinates = read_tsp_file(problem_path)
            tour = read_sol_file(solution_path)
            
            # 计算 TSP 距离
            distance = calculate_tsp_distance(coordinates, tour)
            chunk_distances.append(distance)
        
        # 计算剩余文件的平均距离
        chunk_mean = sum(chunk_distances) / len(chunk_distances)
        total_distances.append(chunk_mean)
        print(f"Chunk {num_chunks + 1}, Mean Distance: {chunk_mean:.6f}")
    
    # 计算总平均距离
    overall_mean = sum(total_distances) / len(total_distances)
    print(f"Overall Mean Distance: {overall_mean:.6f}")

if __name__ == "__main__":
    main()