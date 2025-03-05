import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from peft import get_peft_model, AdapterConfig, PeftType, TaskType
from parser import parse_args

def configure_adapter(model, args):
    """配置Adapter参数"""
    adapter_config = AdapterConfig(
        peft_type=PeftType.ADAPTER,
        task_type=TaskType.SEQ_CLS,
        adapter_size=args.lora_rank * 4,
        r=args.lora_rank,
        bias="none"
    )
    return get_peft_model(model, adapter_config)

def load_dataloader(args):
    """适配器专用数据加载器"""
    # 固定参数设置（论文推荐值）
    MAX_LENGTH = 256  # 输入序列最大长度
    BATCH_SIZE = 32   # 训练batch大小
    
    dataset = load_dataset("snli", split='train')  # 使用完整训练集
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    def preprocess(examples):
        return tokenizer(
            examples['premise'],
            examples['hypothesis'],
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH
        )
    
    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    return (
        DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(dataset, batch_size=BATCH_SIZE),
        len(dataset.features['label'].names),
        "accuracy"
    )

def main():
    # 使用统一的参数解析器
    args = parse_args()
    
    # 初始化模型并应用Adapter
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels
    )
    
    if args.method == "adapter":
        model = configure_adapter(model, args)
    
    # 获取dataloader（实际由其他人实现）
    train_loader, eval_loader, _, _ = load_dataloader(args)
    
    # 基础训练循环
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 优化器仅更新可训练参数
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    for epoch in range(args.num_epochs):
        model.train()
        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"Epoch {epoch+1} completed")

if __name__ == "__main__":
    main()
