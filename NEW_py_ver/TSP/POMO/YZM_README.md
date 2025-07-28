1. 生成.tsp格式数据，数据范围是[0, 1]
运行脚本：
生成数据写入文件夹：/home/yzm/yzm_xp/TSP_precise/database/random01/problems
文件名为dataset_n20_b999.tsp  其中999为number， number从1开始

2. 数据放大：concorde会取整数，需要进行数据放大
运行脚本：/home/yzm/yzm_xp/TSP_precise/concorde/concorde/transition_10000000.py
生成数据写入文件夹/home/yzm/yzm_xp/TSP_precise/database/random01/problems_scaled_up_10000000

3. concorde生成精确解
运行脚本： /home/yzm/yzm_xp/TSP_precise/concorde/concorde/run.py
生成文件写入文件夹： /home/yzm/yzm_xp/TSP_precise/database/random01/concorde_solutions

4. 计算精确解的平均长度
运行脚本： /home/yzm/yzm_xp/TSP_precise/concorde/concorde/calculate_concorde_score.py

5. 比较模型推理解跟精确解： 推理解的均值， 推理序列跟精确序列的一致性
运行脚本： /home/yzm/yzm_xp/TSP_precise/concorde/concorde/check_all.py
依赖：
problems_folder = .tsp文件夹，来自步骤1
concorde_solutions_folder = 精确解的.sol文件夹，来自步骤3
solutions_folder = 模型推理解的.sol文件夹
