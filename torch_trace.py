import torch
import torchvision
import argparse
import os 
import sys
import numpy as np 
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

batch_size_test = 1
total_epoch = 50
base_lr = 0.01
lr_decay_epoch = 5
dev = 'cuda:0' 
work_dir = './trained_models'
log_file = os.path.join(work_dir,'log_test.txt')
test_result_file = 'prediction_result_aa.txt'

criterion = torch.nn.SmoothL1Loss()

def my_load_model(pra_model, pra_path):
	checkpoint = torch.load(pra_path)
	pra_model.load_state_dict(checkpoint['xin_graph_seq2seq_model'])
	print('Successful model loaded from {}'.format(pra_path))
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
	print('Successful data loaded from {}'.format(pra_path))
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


#########

if __name__ == '__main__':
    # set model
    graph_args={'max_hop':2, 'num_node':120}
    model = Model(in_channels=4, graph_args=graph_args, edge_importance_weighting=True)
    
    model.to(dev)
    
    # get model
    pretrained_model_path = 'C:/Ubuntu/GRIP/trained_models/210608_ApolloScape/model_epoch_0049.pt'
    pra_model = my_load_model(model, pretrained_model_path)
    
    # data load
    pra_data_loader = data_loader('C:/Ubuntu/GRIP/test_data_AP.pkl', pra_batch_size=batch_size_test, pra_shuffle=False, pra_drop_last=False, train_val_test='test')
        
    # test model
    pra_model.eval()
    rescale_xy = torch.ones((1,2,1,1)).to(dev) ## 모든 값이 1인 (1, 2, 1, 1) 차원의 tensor 생성
    rescale_xy[:,0] = max_x
    rescale_xy[:,1] = max_y
    all_overall_sum_list = []
    all_overall_num_list = []
    
    # train model using training data
    ## pra_data_loader는 data_loader의 return 결과
    ## N번 (415번) 반복 진행
    for iteration, (ori_data, A, mean_xy) in enumerate(pra_data_loader):
        print(iteration, ori_data.shape, A.shape, mean_xy.shape)
        ### ori_data: (1, C, T, V) 4차원 torch 형태
        ### C = 11: dimension of features [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, object_width, object_height, heading] + [mask]
        ### T = 6: temporal length of the data (history frames(6), future frames(0))
        ### V = 120: maxinum number of objects
        
        ## A: (1, 3, 120, 120) 4차원 torch 형태
    
        ## mean_xy: (1, 2) 2차원 torch 형태
        ### 자차 위치를 나타냄 (전체 데이터의 평균 x, y를 기준으로 자차 위치 표현)
    
        ## rescale은 무시해도 됨, 모두 1이기 때문에 변화 없음
        data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)
        print("ok")
        
        ## history frame (3) 만을 가지는 형태로 변경
        input_data = data[:,:,:history_frames,:] # (1, C, 6->6, V)=(1, 4, 6, 120)
        output_mask = data[:,-1,-1,:] # (1, V)=(1, 120)
        print("ok")
        # print(data.shape, A.shape, mean_xy.shape, input_data.shape)
        ## (1, 4, 6, 120), (1, 3, 120, 120), (1, 2), (1, 4, 6, 120)
    		
        A = A.float().to(dev)
        print("ok")
        # predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=future_frames, pra_teacher_forcing_ratio=0, pra_teacher_location=None)
        # print(predicted)
             
        example = [input_data, A]#, future_frames, 0, None]
        print("ok")
        
        break
    
    
    # trace
    traced_script_module = torch.jit.trace(pra_model, example)
    
    traced_script_module.save("traced_model.pt")



























