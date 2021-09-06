import numpy as np
import torch

class Graph():
	""" The Graph Representation
	How to use:
		1. graph = Graph(max_hop=1)
		2. A = graph.get_adjacency()
		3. A = code to modify A
		4. normalized_A = graph.normalize_adjacency(A)
	"""
	def __init__(self,
				 num_node = 120,
				 max_hop = 1
				 ):
		self.max_hop = max_hop
		self.num_node = num_node 

	def get_adjacency(self, A):
		A = A.detach().cpu().numpy()
		# print("A:", np.shape(A))  
		# compute hop steps
		self.hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
		# print("hop_dis:",np.shape(self.hop_dis))
		transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
		# print("transfer_mat:",np.shape(transfer_mat))
		arrive_mat = (np.stack(transfer_mat) > 0)
		# print("arrive_mat:",np.shape(arrive_mat))
		for d in range(self.max_hop, -1, -1):
			self.hop_dis[arrive_mat[d]] = d

		# compute adjacency
		valid_hop = range(0, self.max_hop + 1)
		adjacency = np.zeros((self.num_node, self.num_node))
		for hop in valid_hop:
			adjacency[self.hop_dis == hop] = 1

		adjacency = torch.from_numpy(adjacency)
		adjacency = adjacency.float().to('cuda:0')

		return adjacency

	def normalize_adjacency(self, A):
		A = A.detach().cpu().numpy()
  
		Dl = np.sum(A, 0)
		num_node = A.shape[0]
		Dn = np.zeros((num_node, num_node))
		for i in range(num_node):
			if Dl[i] > 0:
				Dn[i, i] = Dl[i]**(-1)
		AD = np.dot(A, Dn)

		valid_hop = range(0, self.max_hop + 1)
		A = np.zeros((len(valid_hop), self.num_node, self.num_node))
		for i, hop in enumerate(valid_hop):
			A[i][self.hop_dis == hop] = AD[self.hop_dis == hop]

		A = torch.from_numpy(A)
		A = A.unsqueeze(0)
		A = A.float().to('cuda:0')
		return A


	