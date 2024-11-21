import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
from tqdm import tqdm
from collections import defaultdict
import argparse
import random
from torchvision.utils import save_image

from dataset import CustomDataset, CustomSampler
from osnet import osnet_x1_0
from transform import transform_val, transform_train
from loss import TripletLoss, GlobalTripletLoss, ArcFaceLoss

# 设置命令行参数
def setup_args():
    parser = argparse.ArgumentParser(description='Train and evaluate model')
    parser.add_argument('--train-data-path', default='./dataset/train', type=str, help='Path to training data')
    parser.add_argument('--test-data-path', default='./dataset/query_test', type=str, help='Path to test data')
    parser.add_argument('--batch-size-train', default=99, type=int, help='Batch size for training')
    parser.add_argument('--batch-size-test', default=30, type=int, help='Batch size for testing')
    parser.add_argument('--num-epochs', default=300, type=int, help='Number of epochs to train')
    parser.add_argument('--learning-rate', default=0.001, type=float, help='Learning rate for optimizer')
    parser.add_argument('--sampler-m-per-id', default=3, type=int, help='Number of samples per class ID in each batch')
    parser.add_argument('--checkpoint-path', default='./osnet_x1_0_imagenet.pth', type=str, help='Path to model checkpoint')
    # parser.add_argument('--checkpoint-path', default='./model_last.pth', type=str, help='Path to model checkpoint')
    parser.add_argument('--best-model-path', default='model_best.pth', type=str, help='Path to save the best model')
    parser.add_argument('--last-model-path', default='model_last.pth', type=str, help='Path to save the last model')
    parser.add_argument('--log-file', default='log.txt', type=str, help='Log file path')
    parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
    return parser.parse_args()

def train_one_epoch(epoch_index, model, train_loader, criterion1, criterion2, optimizer, device):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)
    tqdm_iterator = tqdm(train_loader, total=total_batches, desc=f"Epoch {epoch_index + 1}", leave=False)
    for images, labels, _ in tqdm_iterator:
        save_image(images[0], 'test.jpg')
        # print(labels)
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        outputs1, outputs2 = model(images)
        loss1 = criterion1(outputs1, labels)
        loss2 = criterion2(outputs2, labels)*50
        loss = loss1 + loss2
        # loss = criterion2(outputs2, labels)
        loss.backward(retain_graph=True)  # Allow multiple backward passes
        optimizer.step()
        running_loss += loss.item()
        tqdm_iterator.set_description(f"Epoch {epoch_index + 1}, Batch Loss: {loss.item():.4f} (Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f})")
        # tqdm_iterator.set_description(f"Epoch {epoch_index + 1}, Batch Loss: {loss.item():.4f} ")
    avg_loss = running_loss / total_batches
    print(f"Epoch {epoch_index + 1}, Average Loss: {avg_loss:.4f}")
    tqdm_iterator.close()
   

def validate(loader, model, device):
    model.eval()
    all_days = set()
    rank1_accuracy_list = []
    rank3_accuracy_list = []
    rank5_accuracy_list = []

    with torch.no_grad():
        features, pids, days = extract_features(loader, model, device)
        
        # 获取所有的天数
        all_days.update(days.numpy())
        
        for query_day in all_days:
            query_indices = (days == query_day).nonzero(as_tuple=True)[0]
            gallery_indices = (days != query_day).nonzero(as_tuple=True)[0]

            query_features = features[query_indices]
            gallery_features = features[gallery_indices]
            query_pids = pids[query_indices]
            gallery_pids = pids[gallery_indices]

            # 计算分数
            final_scores = defaultdict(float)
            gallery_id_scores = compute_rank5_scores(query_features, gallery_features, query_pids, gallery_pids)
            for key, value in gallery_id_scores.items():
                final_scores[key] += value

            # 找出每个query的分数最高的前五个gallery ids
            query_to_top_galleries = defaultdict(list)
            for (query_id, _), score in final_scores.items():
                query_to_top_galleries[query_id].append((score, _))

            rank1_correct = 0
            rank3_correct = 0
            rank5_correct = 0
            for query_id, scores in query_to_top_galleries.items():
                top_galleries = sorted(scores, reverse=True)[:5]
                top_gallery_ids = [gid for _, gid in top_galleries]
                if query_id == top_gallery_ids[0]:
                    rank1_correct += 1
                if query_id in top_gallery_ids[:3]:
                    rank3_correct += 1
                if query_id in top_gallery_ids:
                    rank5_correct += 1

            rank1_acc = rank1_correct / len(query_to_top_galleries)
            rank3_acc = rank3_correct / len(query_to_top_galleries)
            rank5_acc = rank5_correct / len(query_to_top_galleries)

            rank1_accuracy_list.append(rank1_acc)
            rank3_accuracy_list.append(rank3_acc)
            rank5_accuracy_list.append(rank5_acc)

            print(f'{query_day} Rank-1 Accuracy: {rank1_acc}, Rank-3 Accuracy: {rank3_acc}, Rank-5 Accuracy: {rank5_acc}')
            logging.info(f'{query_day} Rank-1 Accuracy: {rank1_acc}, Rank-3 Accuracy: {rank3_acc}, Rank-5 Accuracy: {rank5_acc}')  # 添加日志记录每次计算的结果

        # 计算平均准确率
        average_rank1_acc = sum(rank1_accuracy_list) / len(all_days)
        average_rank3_acc = sum(rank3_accuracy_list) / len(all_days)
        average_rank5_acc = sum(rank5_accuracy_list) / len(all_days)
        print(f'***Final averaged Rank-1 accuracy: {average_rank1_acc}***')
        print(f'***Final averaged Rank-3 accuracy: {average_rank3_acc}***')
        print(f'***Final averaged Rank-5 accuracy: {average_rank5_acc}***')
        logging.info(f'***Final averaged Rank-1 accuracy: {average_rank1_acc}***')
        logging.info(f'***Final averaged Rank-3 accuracy: {average_rank3_acc}***')
        logging.info(f'***Final averaged Rank-5 accuracy: {average_rank5_acc}***')

    return average_rank1_acc, average_rank3_acc, average_rank5_acc

# def inference(self):
#     self.ckpt.write_log('\n[INFO] Test:')
#     self.model.eval()
#     self.ckpt.add_log(torch.zeros(1, 6))
#     # 在不计算梯度的情况下提取查询集（query set）的特征
#     if self.args.first:
#         with torch.no_grad():
#             # print(self.args.classes_num)
#             output_dir = self.args.first_inference_save
#             os.makedirs(output_dir, exist_ok=True)
#             #得到特征
#             qf, query_ids, query_filepath = self.extract_feature(self.query_loader, self.args)
#             query_feature_dict = dict(zip(query_filepath, qf.numpy()))
#             # print(query_feature_dict)
#             #将特征进行归类
#             categories = defaultdict(list)
#             for feature, q_id, path in zip(qf, query_ids, query_filepath):
#                 category = path.split("day")[-1].split(".")[0]
#                 categories[category].append((feature, q_id, path))
#             #计算两个特征之间的欧式距离，得到欧式距离矩阵
#             distances = cdist(qf, qf, 'euclidean')
#             # distances = cdist(qf, qf, 'cosine')
#             np.fill_diagonal(distances, np.inf)#对角线无限大
#             #生成类别字典
#             categories = {i: [] for i in range(1, self.args.classes_num + 1)}
#             container={i: [] for i in range(200)}

def compute_rank5_scores(query_features, gallery_features, query_pids, gallery_pids):
    # 计算余弦相似度
    normalized_query = query_features / query_features.norm(dim=1, keepdim=True)
    normalized_gallery = gallery_features / gallery_features.norm(dim=1, keepdim=True)
    similarity = torch.mm(normalized_query, normalized_gallery.t())
    # 获取Top-5的索引和对应的余弦相似度值
    top5_values, top5_indices = torch.topk(similarity, 5, largest=True, dim=1)
    
    # 为Top-5分配分数，第一名5分，第五名1分
    scores = torch.tensor([5, 4, 3, 2, 1], dtype=torch.float32)
    gallery_id_scores = defaultdict(float)# 初始化一个默认字典，用于存储每对查询ID和图库ID的累计得分。
    # 遍历每个查询图像和其对应的Top-5图库图像索引。
    for idx, query_id in enumerate(query_pids):
        # idx 是查询特征的索引，query_id 是对应的查询ID
        for rank, gallery_idx in enumerate(top5_indices[idx]):
            # rank 是排名（0-4），gallery_idx 是图库特征的索引。
            gallery_id = gallery_pids[gallery_idx]
            # 为每对查询ID和图库ID添加得分，相同的对可以累计得分。
            gallery_id_scores[(query_id.item(), gallery_id.item())] += scores[rank]
    return gallery_id_scores

def extract_features(loader, model, device):
    features = torch.FloatTensor()
    pids, days = [], []
    for data in loader:
        inputs, pid, day = _parse_data_for_eval(data)
        # print(inputs.shape,pid, day)
        input_img = inputs.to(device)
        outputs = model(input_img)
        outputs = outputs.data.cpu()
        outputs = F.normalize(outputs, p=2, dim=1)  # L2归一化
        features = torch.cat((features, outputs), 0)
        pids.extend(pid.numpy())
        days.extend(day.numpy())
    return features, torch.tensor(pids), torch.tensor(days)

def _parse_data_for_eval(data):
    inputs = data[0]
    pids = data[1]
    days = data[2]
    return inputs, pids, days

# 主函数
def main():
    args = setup_args()

    # 日志配置
    logging.basicConfig(filename=args.log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 数据集和数据加载器
    train_dataset = CustomDataset(args.train_data_path, transform=transform_train, mode='train')
    test_dataset = CustomDataset(args.test_data_path, transform=transform_val, mode='validation')
    sampler = CustomSampler(train_dataset, m=args.sampler_m_per_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, sampler=sampler, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False)

    # 模型初始化和准备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = osnet_x1_0(num_classes=train_dataset.num_classes, pretrained=True)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion1 = nn.CrossEntropyLoss()
    # 假设ArcFaceLoss已经定义
    # criterion2 = ArcFaceLoss(512, train_dataset.num_classes)
    criterion2 = TripletLoss()

    best_accuracy = 0.0

    if args.test_only:
        validate(test_loader, model, device)
    else:
        # 训练和验证循环
        for epoch in range(args.num_epochs):
            train_one_epoch(epoch, model, train_loader, criterion1, criterion2, optimizer, device)
            if (epoch + 1) % 10 == 0:
                average_rank1_accuracy,_,_ = validate(test_loader, model, device)
                if average_rank1_accuracy > best_accuracy:
                    best_accuracy = average_rank1_accuracy
                    torch.save(model.state_dict(), args.best_model_path)
                    logging.info(f"New best model saved with accuracy: {best_accuracy:.2f}%")
                torch.save(model.state_dict(), args.last_model_path)
                logging.info(f"Last model of epoch {epoch} saved.")

if __name__ == '__main__':
    main()
