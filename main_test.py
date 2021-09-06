import argparse
import os 
import sys
import numpy as np 
import torch
import torch.optim as optim
from model import Model
from xin_feeder_baidu import Feeder
from datetime import datetime
import random
import itertools

CUDA_VISIBLE_DEVICES='0'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

def seed_torch(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch()

max_x = 1. 
max_y = 1. 
history_frames = 6 # 3 second * 2 frame/second
future_frames = 6 # 3 second * 2 frame/second

batch_size_train = 64 
batch_size_val = 32
batch_size_test = 1
total_epoch = 50
base_lr = 0.01
lr_decay_epoch = 5
dev = 'cuda:0' 
work_dir = './trained_models'
log_file = os.path.join(work_dir,'log_test.txt')
test_result_file = 'prediction_result_aa.txt'

criterion = torch.nn.SmoothL1Loss()

if not os.path.exists(work_dir):
	os.makedirs(work_dir)
	
def my_load_model(pra_model, pra_path):
	checkpoint = torch.load(pra_path)
	pra_model.load_state_dict(checkpoint['xin_graph_seq2seq_model'])
	print('Successfull loaded from {}'.format(pra_path))
	return pra_model

def data_loader(pra_path, pra_batch_size=128, pra_shuffle=False, pra_drop_last=False, train_val_test='train'):
	feeder = Feeder(data_path=pra_path, graph_args=graph_args, train_val_test=train_val_test)
	loader = torch.utils.data.DataLoader(
		dataset=feeder,
		batch_size=pra_batch_size,
		shuffle=pra_shuffle,
		drop_last=pra_drop_last, 
		num_workers=10,
		)
	return loader
	
def preprocess_data(pra_data, pra_rescale_xy):
	# pra_data: (1, C, T, V)
	# C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]	
	feature_id = [3, 4, 9, 10]

	## 기존 data (1, C, T, V) 에서 C 부분을 position_x, position_y, heading, mask만 추출하여 tensor 복사
	ori_data = pra_data[:,feature_id].detach() ## (1, C, T, V) -> (1, 4, T, V)
	## 복사한 tensor를 재생성
	data = ori_data.detach().clone() ## (1, 4(C), T, V)

	new_mask = (data[:, :2, 1:]!=0) * (data[:, :2, :-1]!=0) 
	data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]).float() * new_mask.float()
	data[:, :2, 0] = 0	

	## small vehicle: 1, big vehicles: 2, pedestrian 3, bicycle: 4, others: 5
	object_type = pra_data[:,2:3]

	data = data.float().to(dev)
	ori_data = ori_data.float().to(dev)
	object_type = object_type.to(dev) #type
	data[:,:2] = data[:,:2] / pra_rescale_xy

	return data, ori_data, object_type

def test_model(pra_model, pra_data_loader):
	# pra_model.to(dev)
	pra_model.eval()
	rescale_xy = torch.ones((1,2,1,1)).to(dev) ## 모든 값이 1인 (1, 2, 1, 1) 차원의 tensor 생성
	rescale_xy[:,0] = max_x
	rescale_xy[:,1] = max_y
	all_overall_sum_list = []
	all_overall_num_list = []

	## output txt 파일에 결과 저장 용도
	with open(test_result_file, 'w') as writer:
		# train model using training data
		## pra_data_loader는 data_loader의 return 결과
		## N번 (415번) 반복 진행
		for iteration, (ori_data, A, mean_xy) in enumerate(pra_data_loader):
 			# print(iteration, ori_data.shape, A.shape, mean_xy.shape)
			
			## ori_data: (1, C, T, V) 4차원 torch 형태
			### C = 11: dimension of features [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, object_width, object_height, heading] + [mask]
			### T = 6: temporal length of the data (history frames(6), future frames(0))
			### V = 120: maxinum number of objects

			## A: (1, 3, 120, 120) 4차원 torch 형태
			###
			###
			###

			## mean_xy: (1, 2) 2차원 torch 형태
			### 자차 위치를 나타냄 (전체 데이터의 평균 x, y를 기준으로 자차 위치 표현)

            ## rescale은 무시해도 됨, 모두 1이기 때문에 변화 없
			data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)
            
            ## history frame (3) 만을 가지는 형태로 변경
			input_data = data[:,:,:history_frames,:] # (1, C, 6->6, V)=(1, 4, 6, 120)
			output_mask = data[:,-1,-1,:] # (1, V)=(1, 120)
 			# print(data.shape, A.shape, mean_xy.shape, input_data.shape)
            ## (1, 4, 6, 120), (1, 3, 120, 120), (1, 2), (1, 4, 6, 120)

            ## 마지막 위치 정보 저장
			ori_output_last_loc = no_norm_loc_data[:,:2,history_frames-1:history_frames,:]
            ## (1, 2, 1, 120)
			# print(ori_output_last_loc.shape)
		
			A = A.float().to(dev)
			predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=future_frames, pra_teacher_forcing_ratio=0, pra_teacher_location=None)
			# print(predicted)
         
			predicted = predicted *rescale_xy
			# print(predicted)                        
            ## (1, 2, 6, 120) * (1, 2, 1, 1)
            
			# print(predicted[:, :, 1].shape)            						
            ## [-2] = 6: 총 6개 프레임에 대해 출력을 더하기 위함
            ## 1 ~ 5까지 5번 반복하는데,
			for ind in range(1, predicted.shape[-2]):
                ## 모든 출력 (속도 관련 정보임)을 더함으로써 최종적으로 얼마나 이동했는 지를 계산하는 과정
                ## 0번 ~ 1번 / (0)+1번 ~ 2번 / (0+1)+2번 ~ 3번 / (0+1+2)+3번 ~ 4번 / (0+1+2+3)+4번 ~ 5번
                ## dim은 더하고 싶은 차원의 index를 넣어줌 (frame에 해당하는 차원)
                ## 각각 frame 0 ~ 6 일 때 이동한 값을 누적하기 때문에, frame 0 ~ 6일 때 이동한 값이 아닌, 0 ~ 6일 때 까지 누적한 값이 됨
				predicted[:,:,ind] = torch.sum(predicted[:,:,ind-1:ind+1], dim=-2)
                ## (1, 2, 120)
			# print(predicted.shape)
            
            ## 예측 값에 마지막 위치 정보를 더해 최종 위치를 업데이트 함
            ## 예측 값은 일종의 속도이기 때문에 위치정보에 속도를 더하는 개념
			predicted += ori_output_last_loc
					
			# 현재 예측한 결과
            ## tensor to cpu numpy 변경하는 과정
			now_pred = predicted.detach().cpu().numpy() # (N, C, T, V)=(N, 2, 6, 120)
			now_pred = np.transpose(now_pred, (0, 2, 3, 1)) # (N, 6, 120, 2)

			# 현재 자차 위치
			now_mean_xy = mean_xy.detach().cpu().numpy() # (N, 2)

			## 현재 들어온 history data
			now_ori_data = ori_data.detach().cpu().numpy() # (N, C, T, V)=(N, 11, 6, 120)
			now_mask = now_ori_data[:, -1, -1, :] # (N, V)=(N, 120)

			now_ori_data = np.transpose(now_ori_data, (0, 2, 3, 1)) # (N, 6, 120, 11)
			
			## N번 반복 => 1번임!
			for n_pred, n_mean_xy, n_data, n_mask in zip(now_pred, now_mean_xy, now_ori_data, now_mask):
				# n_pred: (6, 120, 2)
				# n_mean_xy: (2)
				# n_data: (6, 120, 11)
				# n_mask: (120)
				
				## 주변 object 개수
				num_object = np.sum(n_mask).astype(int)

				# only use the last time of original data for ids (frame_id, object_id, object_type)
				# (6, 120, 11) -> (num_object, 3)
				n_dat = n_data[-1, :num_object, :3].astype(int)
				for time_ind, n_pre in enumerate(n_pred[:, :num_object], start=1):
					# time_ind: (120, 2) -> (n, 2)
					
					for info, pred in zip(n_dat, n_pre+n_mean_xy):
						information = info.copy()
						information[0] = information[0] + time_ind
						result = ' '.join(information.astype(str)) + ' ' + ' '.join(pred.astype(str)) + '\n'
						# frame_id / object_id / object_type             + X / Y
						writer.write(result)

def run_test(pra_model, pra_data_path):
	## 학습된 .pt 형식의 모델(pra_model)을 불러옴
	loader_test = data_loader(pra_data_path, pra_batch_size=batch_size_test, pra_shuffle=False, pra_drop_last=False, train_val_test='test')

	## 모델과 test data(pra_data_path)를 넣어 test 결과 출력
	test_model(pra_model, loader_test)

if __name__ == '__main__':
	## graph 설정 관련 argument
	graph_args={'max_hop':2, 'num_node':120}

	## model 생성
	## model.py 클래스를 불러와서 객체 생성
	model = Model(in_channels=4, graph_args=graph_args, edge_importance_weighting=True)

	## GPU 학습을 위해 초기화된 모델에 to(dev)를 통해 Cuda에 최적화된 모델로 변환
	## dev = 'cuda:0'
	model.to(dev)
	
	##### test #####
	## 학습 된 모델 중 원하는 모델 선택, 이를 기반으로 test 진행
	pretrained_model_path = 'C:/Ubuntu/GRIP/trained_models/210608_ApolloScape/model_epoch_0049.pt'
	model = my_load_model(model, pretrained_model_path)
	print("Load Model Finished!")
	run_test(model, 'C:/Ubuntu/GRIP/test_data_AP.pkl')
	
		
		

