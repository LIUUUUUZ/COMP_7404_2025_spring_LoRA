import os
import torch
from transformers import AutoModelForSequenceClassification
from adapters import AutoAdapterModel, AdapterConfig
from arg_parser import parse_args

# 从 main.py 中导入复用的函数
from main import (
	set_seed,
	load_and_preprocess_data,
	compute_metrics,
	train,
	evaluate
)


def load_model(args, num_labels):
	"""根据方法加载模型, 并根据需要应用Adapter."""
	model = AutoAdapterModel.from_pretrained(args.model_name)

	model.add_classification_head(
		head_name="task_head",
		num_labels=num_labels
	)
	# 显式解冻分类头
	for param in model.heads["task_head"].parameters():
		param.requires_grad = True

	adapter_config = AdapterConfig.load(
		"pfeiffer",
		reduction_factor=768 // args.lora_rank,
		layers_to_transform=list(range(6, 12)),
		target_modules=["attention.self.query", "attention.self.key", "attention.self.value", "output.dense"]
	)

	model.add_adapter("adapter", adapter_config)
	model.train_adapter("adapter")
	model.set_active_adapters("adapter")

	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	all_params = sum(p.numel() for p in model.parameters())
	print(f"Adapter配置:")
	print(f"Reduction Factor: {768 // args.lora_rank}")
	print(f"可训练参数: {trainable_params} ({trainable_params / all_params:.2%})")

	return model


def run_experiments(args):
	"""根据参数运行实验."""
	set_seed(args.seed)
	os.makedirs(args.output_dir, exist_ok=True)
	train_dataloader, eval_dataloader, num_labels, metric_name = load_and_preprocess_data(args)

	if args.mode == "train":
		model = load_model(args, num_labels)
		train_time = train(args, model, train_dataloader, eval_dataloader, metric_name)
		print(f"总训练时间: {train_time:.2f} 秒")

		# 评估模型（用于预测时间）
		_, predict_time = evaluate(args, model, eval_dataloader, metric_name, predict=True)
		print(f"预测 {args.test_iters} 批次的时间: {predict_time:.2f} 秒")

	elif args.mode == "eval":
		model_path = os.path.join(
			args.output_dir,
			f"{args.dataset}_{args.method}_rank{args.lora_rank if args.method == 'lora' else ''}"
		)
		model = AutoAdapterModel.from_pretrained(model_path)

		# 评估模型
		metrics, eval_time = evaluate(args, model, eval_dataloader, metric_name)
		print(f"评估结果: {metrics}")
		print(f"评估时间: {eval_time:.2f} 秒")

		# 测量预测时间
		_, predict_time = evaluate(args, model, eval_dataloader, metric_name, predict=True)
		print(f"预测 {args.test_iters} 批次的时间: {predict_time:.2f} 秒")


def main():
	"""主函数."""
	args = parse_args()

	if args.time_count:
		start_time = time.time()

	run_experiments(args)

	if args.time_count:
		total_time = time.time() - start_time
		print(f"总执行时间: {total_time:.2f} 秒")


if __name__ == "__main__":
	main()
