import os
import time
import torch
import wandb
import argparse
import logging
from datetime import datetime
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["WANDB_MODE"] = "run"


class MLP(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 5),
        )

    def forward(self, x):
        x = self.network(x)
        return x


class RandomDataset(Dataset):
    def __init__(self, length):
        self.len = length
        self.data = torch.stack([torch.ones(5), torch.ones(5)*2,
                                 torch.ones(5)*3,torch.ones(5)*4,
                                 torch.ones(5)*5,torch.ones(5)*6,
                                 torch.ones(5)*7,torch.ones(5)*8,
                                 torch.ones(5)*9, torch.ones(5)*10,
                                 torch.ones(5)*11,torch.ones(5)*12,
                                 torch.ones(5)*13,torch.ones(5)*14,
                                 torch.ones(5)*15,torch.ones(5)*16]).to('cuda')
        
        self.label = torch.stack([torch.zeros(5), torch.zeros(5)*2,
                                torch.zeros(5)*3,torch.zeros(5)*4,
                                torch.zeros(5)*5,torch.zeros(5)*6,
                                torch.zeros(5)*7,torch.zeros(5)*8,
                                torch.zeros(5)*9, torch.zeros(5)*10,
                                torch.zeros(5)*11,torch.zeros(5)*12,
                                torch.zeros(5)*13,torch.zeros(5)*14,
                                torch.zeros(5)*15,torch.zeros(5)*16]).to('cuda')

    def __getitem__(self, index):

        return [self.data[index], self.label[index]]

    def __len__(self):

        return self.len
    
    def collate_batch(self, batch):
        """
        实现一个自定义的batch拼接函数，这个函数将会被传递给DataLoader的collate_fn参数
        """
        data = torch.stack([x[0] for x in batch])
        label = torch.stack([x[1] for x in batch])
        return [data, label]


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    print("rank: ", rank)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')  # 只有当rank=0时，才会输出info信息
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--batch_size', type=int, default=2, required=False, help='batch size for training')
    parser.add_argument('--train_epochs', type=int, default=100, required=False, help='number of epochs to train for')
    parser.add_argument('--train_lr', type=int, default=1e-3, required=False, help='training learning rate')
    parser.add_argument('--loader_num_workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--training', type=bool, default=True, help='training or testing mode')
    parser.add_argument('--without_sync_bn', type=bool, default=False, help='whether to use Synchronization Batch Normaliation')
    parser.add_argument('--output_dir', type=bool, default="/home/", help='output directory for saving model and logs')
    
    args = parser.parse_args()

    return args


def main():
    args = parse_config()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # 初始化分布式参数
    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ['LOCAL_RANK'])  # 每个卡上运行一个程序(系统自动分配local_rank)
    # local_rank = 0  # 在第一张卡运行多个程序
    torch.cuda.set_device(local_rank % num_gpus)  # 设置当前卡的device
    
    # 初始化分布式
    dist.init_process_group(
        backend="nccl",  # 指定后端通讯方式为nccl
        # init_method='tcp://localhost:23456',
        rank=local_rank,  # rank是指当前进程的编号
        world_size=num_gpus  # worla_size是指总共的进程数
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    os.makedirs(args.output_dir, exist_ok=True)

    # 配置日志
    logger_file = "/home/log.txt"
    logger = create_logger(log_file=logger_file, rank=rank)
    logger.info("rank: %d, world_size: %d" % (rank, world_size))
    logger.info("local_rank: %d, num_gpus: %d" % (local_rank, num_gpus))

    wandb_logger = logging.getLogger("wandb")
    wandb_logger.setLevel(logging.WARNING)
    wandb_config = wandb.init(
        project="test-project", 
        name=str(datetime.now().strftime('%Y.%m.%d-%H.%M.%S'))+f"rank-{rank}",
        dir=args.output_dir,
    )

    # 加载并切分数据集
    dataset = RandomDataset(16)
    if args.training:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, world_size, rank, shuffle=False)
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        collate_fn=dataset.collate_batch, num_workers=args.loader_num_workers
    )

    # 加载模型
    model = MLP().to(device)
    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of parameters: {total_params}")
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank % torch.cuda.device_count()])
    if not args.without_sync_bn:  # 不同卡之间进行同步批量正则化
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.train()

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.train_lr, weight_decay=0)
    # 定义学习率调度器
    milestones = [10, 30, 70]  # 在这些 epoch 后降低学习率
    gamma = 0.5  # 学习率降低的乘数因子
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # 训练模型
    for iter in range(args.train_epochs):
        sampler.set_epoch(iter)
        logger.info("-------------epoch %d start----------------" % iter)
        for _, batch in enumerate(data_loader):
            data, label = batch[0], batch[1]
            data=data.to(device)
            label=label.to(device)
            output = model(data)
            loss = (label - output).pow(2).sum()

            optimizer.zero_grad()
            loss.backward()
            logger.info("loss: %s" % str(loss))
            wandb_config.log({"loss": loss})
            optimizer.step()
            scheduler.step()
        
        logger.info("-------------epoch %d end----------------" % iter)
        time.sleep(10)

if __name__ == "__main__":
    main()
 
