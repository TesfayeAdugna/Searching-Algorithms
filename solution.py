from math import atan2, cos, radians, sin, sqrt
import graphimplementation as gi
from collections import deque
import heapq
import timeit
import matplotlib.pyplot as plt

class Solution:

    def __init__(self):
        self.g = gi.Graph()
        self.Earth_Radius = 6373.0
        self.time_of_test = 10
        self.heuristic_data = {}
        self.count = 0

    # This function creates a graph based on the edges and
    # heuristic data given.
    def create_graph(self, graph_file, heuristic_file):
        with open(graph_file) as file:
            for line in file:
                connection = line.split()
                if connection[0] not in self.g.verticies:
                    node1 = gi.Node(connection[0])
                    self.g.add_node(node1)
                if connection[1] not in self.g.verticies:
                    node2 = gi.Node(connection[1])
                    self.g.add_node(node2)
                self.g.add_edge(self.g.verticies[connection[0]], self.g.verticies[connection[1]],connection[2])

        with open(heuristic_file) as file:
            for line in file:
                data = line.split()
                self.heuristic_data[data[0]] = [radians(float(data[1])),radians(float(data[2]))]

    # This function accepts two indexes and returns nodes at that index
    def find_nodes(self,ind1,ind2):
        for i,key in enumerate(self.g.verticies):
            if i == ind1:
                start = self.g.verticies[key]
            if i == ind2:
                end = self.g.verticies[key]
                break
        return [start,end]

    # BFS graph traversal algorithm to find level difference 
    # between start and end nodes.
    def bfs(self, start, end):
        count = 0
        queue = deque([start])
        visited = set()
        while queue:
            count+=1
            for _ in range(len(queue)):
                temp = queue.popleft()
                visited.add(temp.name)
                for nodes in temp.edge_list:
                    if nodes.name not in visited:
                        queue.append(nodes)
                        if nodes.name == end.name:
                            return count
        return count

    # DFS graph traversal algorithm
    def dfs(self, start, end, dfs_visited):
        dfs_visited.add(start.name)
        if start.name == end.name:
            self.count = len(dfs_visited)
            return
        for nodes in start.edge_list:
            if nodes.name not in dfs_visited:
                self.dfs(nodes, end, dfs_visited)
        return self.count

    # Djikstra shortest path algorithm implementation
    # this function accepts start node and returns shortest distance 
    # to reach each node starting from given start node.
    def djikstra(self, start):
        dis_start = {}
        previous = {}
        for node_name in self.g.verticies:
            dis_start[node_name] = float("inf")
            previous[node_name] = start.name
        dis_start[start.name] = 0
        unvisited = [[0, start.name]]
        heapq.heapify(unvisited)
        visited = set()
        while unvisited:
            temp = heapq.heappop(unvisited)
            visited.add(temp[1])
            for node in self.g.verticies[temp[1]].edge_list:
                if node.name not in visited:
                    new_dis = dis_start[temp[1]] + int(self.g.edges[(node.name, temp[1])].weight)
                    if new_dis < dis_start[node.name]:
                        dis_start[node.name] = new_dis
                        previous[node.name] = temp[1]
                        heapq.heappush(unvisited, [dis_start[node.name], node.name])
        return [dis_start, previous, len(visited)]

    # a function to calculate heuristic distance from every node
    # to the end node. this function accepts end node and returns
    # a dictionary which contains all node and their calculated 
    # distance from end node.
    def calc_heuristic_dis(self, end):
        end_pos = self.heuristic_data[end.name]
        heuristic_dis = {}
        for city in self.heuristic_data:
            temp = self.heuristic_data[city]
            Haversine = sin((temp[0]-end_pos[0]) / 2)**2 + cos(temp[0]) * cos(end_pos[0]) * sin((temp[1]-end_pos[1]) / 2)**2
            c = 2 * atan2(sqrt(Haversine), sqrt(1 - Haversine))
            heuristic_dis[city] = self.Earth_Radius * c
        return heuristic_dis

    # A* search algorithm implementation.
    # this function accepts start and end node, and returns the
    # first shortest path found. since A* search chooses the path 
    # greedly we might not get the shortest paths.
    def Astarsearch(self, start, end):
        heuristic_dis = self.calc_heuristic_dis(end)
        f = {}
        dis_start = {}
        previous = {}
        for node_name in self.g.verticies:
            dis_start[node_name] = float("inf")
            previous[node_name] = start.name
            f[node_name] = float("inf")
        dis_start[start.name] = 0
        f[start.name] = heuristic_dis[start.name] + dis_start[start.name]
        unvisited = [[heuristic_dis[start.name], start]]
        heapq.heapify(unvisited)
        visited = set()
        flag = False
        while unvisited:
            temp = heapq.heappop(unvisited)
            visited.add(temp[1])
            for node in temp[1].edge_list:
                if node not in visited:
                    new_dis = dis_start[temp[1].name] + int(self.g.edges[(node.name, temp[1].name)].weight)
                    temp_dis = heuristic_dis[node.name] + new_dis
                    if temp_dis < f[node.name]:
                        # if temp_dis < f[node.name]:
                        dis_start[node.name] = new_dis
                        previous[node.name] = temp[1].name
                        f[node.name] = temp_dis
                        heapq.heappush(unvisited, [f[node.name], node])
                    if node.name == end.name:
                        flag = True
                        break
            if flag:
                break
        # finding the path to reach end starting from start node
        path = []
        while end.name != start.name:
            path.append(end.name)
            end = self.g.verticies[previous[end.name]]
        path.append(start.name)
        shortest_path = path[::-1]
        return [dis_start,shortest_path,len(shortest_path)]

    # benchmarking the above four searching algorithms.
    def benchmark(self):
        n = len(self.g.verticies)
        num_of_node_connection = n * (n-1)
        Bench_collection = []
        # benchmarking BFS
        bfs_total_time = 0
        for start_node in self.g.verticies:
            for end_node in self.g.verticies:
                if start_node!=end_node:
                    bfs_total_time += timeit.timeit(lambda: self.bfs(self.g.verticies[start_node], self.g.verticies[end_node]), number = self.time_of_test)
        bfs_benchmark = bfs_total_time/(self.time_of_test*num_of_node_connection)
        Bench_collection.append(bfs_benchmark)

        # benchmarking DFS
        dfs_total_time = 0
        for start_node in self.g.verticies:
            for end_node in self.g.verticies:
                if start_node!=end_node:
                    dfs_visited = set()
                    self.count = 0
                    dfs_total_time += timeit.timeit(lambda: self.dfs(self.g.verticies[start_node], self.g.verticies[end_node], dfs_visited), number = self.time_of_test)
        dfs_benchmark = dfs_total_time/(self.time_of_test*num_of_node_connection)
        Bench_collection.append(dfs_benchmark)

        # benchmarking Djikstra
        djikstra_total_time = 0
        total_loop = 20
        for start_node in self.g.verticies:
            djikstra_total_time += timeit.timeit(lambda: self.djikstra(self.g.verticies[start_node]), number = self.time_of_test)
        djikstra_benchmark = djikstra_total_time/(self.time_of_test*total_loop)
        Bench_collection.append(djikstra_benchmark)

        # benchmarking A* search
        Astar_total_time = 0
        for start_node in self.g.verticies:
            for end_node in self.g.verticies:
                if start_node != end_node:
                    Astar_total_time += timeit.timeit(lambda: self.Astarsearch(self.g.verticies[start_node],self.g.verticies[end_node]), number = self.time_of_test)
        Astar_benchmark = Astar_total_time/(self.time_of_test*num_of_node_connection)
        Bench_collection.append(Astar_benchmark)

        return Bench_collection

    # distance benchmark
    def benchmark_distance(self):
        n = len(self.g.verticies)
        num_of_node_connection = n * (n-1)
        Bench_collection = []
        # benchmarking BFS
        bfs_count = 0
        for start_node in self.g.verticies:
            for end_node in self.g.verticies:
                if start_node!=end_node:
                    bfs_count += self.bfs(self.g.verticies[start_node], self.g.verticies[end_node])
        bfs_benchmark = bfs_count/(num_of_node_connection)
        Bench_collection.append(bfs_benchmark)

        # benchmarking DFS
        dfs_count = 0
        for start_node in self.g.verticies:
            for end_node in self.g.verticies:
                if start_node!=end_node:
                    dfs_visited = set()
                    self.count = 0
                    dfs_count += self.dfs(self.g.verticies[start_node], self.g.verticies[end_node], dfs_visited)
        dfs_benchmark = dfs_count/(num_of_node_connection)
        Bench_collection.append(dfs_benchmark)

        # benchmarking Djikstra
        djikstra_count = 0
        total_loop = 20
        for start_node in self.g.verticies:
            djikstra_count += self.djikstra(self.g.verticies[start_node])[2]
        djikstra_benchmark = djikstra_count/(total_loop)
        Bench_collection.append(djikstra_benchmark)

        # benchmarking A* search
        Astar_count = 0
        for start_node in self.g.verticies:
            for end_node in self.g.verticies:
                if start_node != end_node:
                    Astar_count += self.Astarsearch(self.g.verticies[start_node],self.g.verticies[end_node])[2]
        Astar_benchmark = Astar_count/(num_of_node_connection)
        Bench_collection.append(Astar_benchmark)

        return Bench_collection

    # drawing the searching algorithms benchmark graph
    def draw_average_time_graph(self):
        # y = self.benchmark()
        y = self.benchmark_distance()
        x = [1, 2, 3, 4]
        tick_label = ['BFS', 'DFS', 'Dijkstra', 'A* search']
        plt.title("search algorithms benchmarking")
        plt.bar(x, y, tick_label = tick_label,width = 0.8, color = ['green', 'red', 'blue', 'black'])
        plt.show()



    """
    Group work starts here. Members...
          Name                   Id               Section
    1. Tesfaye Adugna        UGR/4709/12             2
    2. Kenna Tefera          UGR/0317/12             2
    """


    # helper function to calculate total weights of the edges.
    def total_weight_calc(self):
        total_w = 0
        for edge in self.g.edges:
            total_w += int(self.g.edges[edge].weight)
        return total_w
    
    # calculating the degree centrality
    def degree_centrality(self):
        total_weight = self.total_weight_calc()
        degree_cent = {}
        for node in self.g.verticies:
            single_wight = 0
            for edge in self.g.verticies[node].edge_list:
                single_wight += int(self.g.edges[(node,edge.name)].weight)
            CD = single_wight/(total_weight)
            degree_cent[node] = CD
        return degree_cent

    # calculating the closeness centrality using djikstra
    def closeness_centrality_Dj(self):
        total_weight = self.total_weight_calc()
        clos_cent = {}
        for node1 in self.g.verticies:
            shortest_paths = self.djikstra(self.g.verticies[node1])[0]
            temp = 0
            for weights in shortest_paths:
                temp+=shortest_paths[weights]
            CC = (total_weight)/temp
            clos_cent[node1] = CC
        return clos_cent

    # calculating closeness centrality using A* search
    def closeness_centrality_As(self):
        total_weight = self.total_weight_calc()
        clos_cent = {}
        for node1 in self.g.verticies:
            temp = 0
            for node2 in self.g.verticies:
                if node1 != node2:
                    shortest_paths = self.Astarsearch(self.g.verticies[node1], self.g.verticies[node2])[0]
                    temp += shortest_paths[node2]
            CC = (total_weight)/temp
            clos_cent[node1] = CC
        return clos_cent

    # betweenness centrality using Djikstra
    def betweenness_centrality_Dj(self):
        total_connections = 380
        bet_cent = {}
        for start in self.g.verticies:
            temp = 0
            for node1 in self.g.verticies:
                if node1 != start:
                    prev_nodes = self.djikstra(self.g.verticies[node1])[1]
                    for node2 in prev_nodes:
                        if node2 != start:
                            temp_n = node2
                            while temp_n != node1:
                                if prev_nodes[temp_n] == start or temp_n == start:
                                    temp+=1
                                    break
                                temp_n = prev_nodes[temp_n]
            bet_cent[start] = temp/total_connections
        return bet_cent

    # betweenness centrality using A* search
    def betweenness_centrality_As(self):
        total_connections = 380
        bet_cent = {}
        for start in self.g.verticies:
            temp = 0
            for node1 in self.g.verticies:
                for node2 in self.g.verticies:
                    if node1 != node2 and node1!=start and node2!=start:
                        shortest = self.Astarsearch(self.g.verticies[node1], self.g.verticies[node2])[1]
                        if start in shortest:
                            temp+=1
            bet_cent[start] = temp/total_connections
        return bet_cent

    # benchmarking closeness centrality
    def benchmark_closeness_centrality(self):
        clos_cent_Dj = self.closeness_centrality_Dj()
        clos_cent_As = self.closeness_centrality_As()
        table_data = [['City names','Closeness using Dj','Closeness using A*','difference']]
        for clos1, clos2 in zip(clos_cent_Dj, clos_cent_As):
            temp = []
            temp.append(clos1)
            temp.append(clos_cent_Dj[clos1])
            temp.append(clos_cent_As[clos2])
            temp.append(abs(clos_cent_Dj[clos1]-clos_cent_As[clos2]))
            table_data.append(temp)
        fig, ax = plt.subplots()
        table = ax.table(cellText=table_data, loc='center')
        table.scale(1.25, 0.70)
        ax.axis('off')
        plt.show()

    # benchmarking betweenness centrality
    def benchmark_betweenness_centrality(self):
        bet_cent_Dj = self.betweenness_centrality_Dj()
        bet_cent_As = self.betweenness_centrality_As()
        table_data = [['City names','Betweenness using Dj','Betweenness using A*','difference']]
        for clos1, clos2 in zip(bet_cent_Dj, bet_cent_As):
            temp = []
            temp.append(clos1)
            temp.append(bet_cent_Dj[clos1])
            temp.append(bet_cent_As[clos1])
            temp.append(abs(bet_cent_Dj[clos1]-bet_cent_As[clos1]))
            table_data.append(temp)
        fig, ax = plt.subplots()
        table = ax.table(cellText=table_data, loc='center')
        table.set_fontsize(36)
        table.scale(1.25,1.2)
        ax.axis('off')
        plt.show()

    # benchmarking all types of centralities in one table
    def benchmark_centrality(self):
        degree_cent = self.degree_centrality()
        clos_cent_Dj = self.closeness_centrality_Dj()
        clos_cent_As = self.closeness_centrality_As()
        bet_cent_Dj = self.betweenness_centrality_Dj()
        bet_cent_As = self.betweenness_centrality_As()
        table_data = [['City names','Degree Centrality','Closeness using Dj','Closeness using A*','Betweenness using Dj','Betweenness using A*']]
        for clos0, clos1, clos2, clos3, clos4 in zip(degree_cent, clos_cent_Dj,clos_cent_As, bet_cent_Dj, bet_cent_As):
            temp = []
            temp.append(clos1)
            temp.append(degree_cent[clos1])
            temp.append(clos_cent_Dj[clos1])
            temp.append(clos_cent_As[clos1])
            temp.append(bet_cent_Dj[clos1])
            temp.append(bet_cent_As[clos1])
            table_data.append(temp)
        fig, ax = plt.subplots()
        table = ax.table(cellText=table_data, loc='center')
        table.scale(1.25,1.2)
        ax.axis('off')
        plt.show()


s = Solution()
graph_file_and_heuristic = {'./graph_files/texts1x.txt':'./heuristic_data/lat_long_pairs1x.txt', './graph_files/texts2x.txt':'./heuristic_data/lat_long_pairs2x.txt', './graph_files/texts3x.txt':'./heuristic_data/lat_long_pairs3x.txt', './graph_files/texts4x.txt':'./heuristic_data/lat_long_pairs4x.txt', './graph_files/texts5x.txt':'./heuristic_data/lat_long_pairs5x.txt'}
s.create_graph('./graph_files/texts1x.txt', graph_file_and_heuristic['./graph_files/texts1x.txt'])
start,end = s.find_nodes(1,8)
