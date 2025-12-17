# import json
# import re
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
#
#
# def load_jsonl(file_path):
#     """加载JSONL格式的测试数据集"""
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data.append(json.loads(line.strip()))
#     return data
#
#
# def load_model(base_model_path, checkpoint_path=None, device='cuda'):
#     """加载基础模型和可选的微调CheckPoint（LoRA适配器）"""
#     tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
#
#     # 修复：使用 dtype 替代已废弃的 torch_dtype
#     model = AutoModelForCausalLM.from_pretrained(
#         base_model_path,
#         dtype=torch.bfloat16,  # 修正参数名
#         trust_remote_code=True
#     ).eval().to(device)
#
#     if checkpoint_path:  # 如果提供了微调CheckPoint路径，加载LoRA权重
#         model = PeftModel.from_pretrained(model, checkpoint_path).to(device)
#     return model, tokenizer
#
#
# def safe_json_parse(text, default=None):
#     """
#     解析模型输出的JSON，避免格式错误
#     参数:
#         text: 要解析的JSON字符串
#         default: 解析失败时返回的默认值
#     """
#     if not text or not text.strip():
#         return default
#
#     # 清理可能存在的Markdown代码块标记
#     text_clean = re.sub(r'^```json\s*|\s*```$', '', text.strip())
#
#     # 预处理：移除非法控制字符和零宽空格
#     text_clean = re.sub(r'[\x00-\x1F]', ' ', text_clean)  # 控制字符
#     text_clean = re.sub(r"[\u200B-\u200D\uFEFF]", "", text_clean)  # 零宽空格
#
#     # 修复常见格式错误
#     text_clean = re.sub(r",\s*}", "}", text_clean)  # 尾随逗号
#     text_clean = re.sub(r",\s*]", "]", text_clean)  # 数组尾随逗号
#
#     try:
#         return json.loads(text_clean)
#     except json.JSONDecodeError as e:
#         print(f"JSON解析错误: {e.msg}")
#         print(f"错误位置: {e.pos}, 上下文: {text_clean[max(0, e.pos - 30):e.pos + 30]}")
#
#         # 尝试宽松模式解析
#         try:
#             return json.loads(text_clean, strict=False)
#         except:
#             # 作为最后手段尝试提取JSON对象
#             try:
#                 # 尝试匹配 {...} 模式
#                 match = re.search(r'\{[^}]*\}', text_clean)
#                 if match:
#                     return json.loads(match.group())
#                 return default
#             except:
#                 return default
#
#
# def predict(model, tokenizer, instruction, input_text, device='cuda'):
#     """对单条样本进行预测，返回详细的预测信息"""
#     # 组合指令和输入文本作为完整Prompt
#     prompt = f"{instruction}\n\n{input_text}"
#
#     # 使用apply_chat_template处理输入
#     messages = [{"role": "user", "content": prompt}]
#
#     # 处理tokenizer输入
#     if hasattr(tokenizer, 'apply_chat_template') and tokenizer.apply_chat_template is not None:
#         try:
#             inputs = tokenizer.apply_chat_template(
#                 messages,
#                 add_generation_prompt=True,
#                 tokenize=True,
#                 return_tensors="pt",
#                 return_dict=True,
#                 enable_thinking=False
#             )
#         except Exception as e:
#             print(f"apply_chat_template失败，使用直接编码: {e}")
#             inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
#     else:
#         inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
#
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#
#     # 生成参数设置
#     gen_kwargs = {
#         "max_new_tokens": 512,
#         "do_sample": False,
#         "pad_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id
#     }
#
#     try:
#         with torch.no_grad():
#             outputs = model.generate(**inputs, **gen_kwargs)
#
#             # 提取生成的文本
#             if inputs['input_ids'].shape[1] < outputs.shape[1]:
#                 response = tokenizer.decode(
#                     outputs[0][inputs['input_ids'].shape[1]:],
#                     skip_special_tokens=True
#                 )
#             else:
#                 response = ""
#
#         # 保存原始响应
#         raw_response = response
#
#         # 尝试解析JSON
#         parsed = safe_json_parse(response, {'is_fraud': False})
#
#         # 添加对列表的处理
#         if isinstance(parsed, list):
#             # 如果是列表，尝试获取第一个元素（假设列表中包含字典）
#             if parsed and isinstance(parsed[0], dict):
#                 prediction = bool(parsed[0].get('is_fraud', False))
#                 parsed_dict = parsed[0]
#             else:
#                 prediction = False
#                 parsed_dict = {}
#         elif isinstance(parsed, dict):
#             # 如果是字典，正常处理
#             prediction = bool(parsed.get('is_fraud', False))
#             parsed_dict = parsed
#         else:
#             # 其他类型，返回False
#             prediction = False
#             parsed_dict = {}
#
#         return {
#             'prediction': prediction,
#             'raw_response': raw_response,
#             'parsed_dict': parsed_dict,
#             'success': parsed is not None
#         }
#
#     except Exception as e:
#         print(f"预测过程中出错: {e}")
#         return {
#             'prediction': False,
#             'raw_response': '',
#             'parsed_dict': {},
#             'success': False,
#             'error': str(e)
#         }
#
#
# def evaluate_model(test_data_path, base_model_path, checkpoint_path=None, device='cuda'):
#     """主评估函数：加载数据/模型，预测并计算指标"""
#     # 1. 加载测试数据
#     try:
#         test_data = load_jsonl(test_data_path)
#         print(f"成功加载 {len(test_data)} 条测试数据")
#     except Exception as e:
#         print(f"加载测试数据失败: {e}")
#         return 0, 0, 0, 0, 0, 0, 0
#
#     # 2. 加载模型
#     try:
#         model, tokenizer = load_model(base_model_path, checkpoint_path, device)
#         print("模型加载成功")
#     except Exception as e:
#         print(f"模型加载失败: {e}")
#         return 0, 0, 0, 0, 0, 0, 0
#
#     # 3. 收集真实标签和预测标签
#     true_labels = []
#     pred_labels = []
#     detailed_results = []  # 存储详细结果
#
#     for i, sample in enumerate(test_data):
#         try:
#             instruction = sample.get("instruction", "")
#             input_text = sample["input"]
#             true_label = sample["label"]
#
#             # 模型预测
#             result = predict(model, tokenizer, instruction, input_text, device)
#             pred_label = result['prediction']
#             true_labels.append(true_label)
#             pred_labels.append(pred_label)
#
#             # 存储详细结果
#             detailed_results.append({
#                 'index': i + 1,
#                 'input_text': input_text[:200] + "..." if len(input_text) > 200 else input_text,  # 截取前200字符
#                 'true_label': true_label,
#                 'prediction': pred_label,
#                 'raw_response': result['raw_response'],
#                 'parsed_dict': result['parsed_dict'],
#                 'success': result.get('success', False)
#             })
#
#         except Exception as e:
#             print(f"第{i + 1}个样本处理失败: {e}")
#             # 预测失败时使用默认值False
#             true_labels.append(true_label)
#             pred_labels.append(False)
#
#             detailed_results.append({
#                 'index': i + 1,
#                 'input_text': input_text[:200] + "..." if len(input_text) > 200 else input_text,
#                 'true_label': true_label,
#                 'prediction': False,
#                 'raw_response': "",
#                 'parsed_dict': {},
#                 'success': False,
#                 'error': str(e)
#             })
#
#         # 打印进度（每10%）
#         if (i + 1) % max(1, len(test_data) // 10) == 0:
#             print(f"进度: {i + 1}/{len(test_data)} ({((i + 1) / len(test_data)) * 100:.1f}%)")
#
#     # 4. 计算指标
#     if not true_labels:
#         print("没有有效数据可用于计算指标")
#         return 0, 0, 0, 0, 0, 0, 0
#
#     try:
#         # 使用sklearn直接计算TP、FP、TN、FN
#         cm = confusion_matrix(true_labels, pred_labels, labels=[True, False])
#         TN, FP, FN, TP = cm.ravel()
#
#         # 计算精确率、召回率、F1
#         precision = precision_score(true_labels, pred_labels, pos_label=True, zero_division=0)
#         recall = recall_score(true_labels, pred_labels, pos_label=True, zero_division=0)
#         f1 = f1_score(true_labels, pred_labels, pos_label=True, zero_division=0)
#
#     except Exception as e:
#         print(f"指标计算失败: {e}")
#         return 0, 0, 0, 0, 0, 0, 0
#
#     # 5. 输出结果
#     print("\n" + "=" * 80)
#     print("评估结果")
#     print("=" * 80)
#     print(f"TP (True Positive):  {TP}")
#     print(f"FP (False Positive): {FP}")
#     print(f"TN (True Negative):  {TN}")
#     print(f"FN (False Negative): {FN}")
#     print("-" * 80)
#     print(f"精确率 (Precision): {precision:.4f}")
#     print(f"召回率 (Recall):    {recall:.4f}")
#     print(f"F1分数:            {f1:.4f}")
#
#     # 6. 输出前3个样本的详细信息
#     print("\n" + "=" * 80)
#     print("前3个样本的详细预测信息")
#     print("=" * 80)
#     #666
#     for i in range(min(3, len(detailed_results))):
#         result = detailed_results[i]
#         print(f"\n样本 {result['index']}:")
#         print(f"输入文本: {result['input_text']}")
#         print(f"真实标签: {result['true_label']}")
#         print(f"预测标签: {result['prediction']}")
#         print(f"预测成功: {result['success']}")
#         print(f"原始模型输出: {result['raw_response'][:1000]}{'...' if len(result['raw_response']) > 1000 else ''}")
#         print(f"解析后的字典: {result['parsed_dict']}")
#         if 'error' in result:
#             print(f"错误信息: {result['error']}")
#         print("-" * 40)
#
#     return TP, FP, TN, FN, precision, recall, f1
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 配置路径和设备
#     test_data_path = "/home/sxk1/EI/Data/4Gambling/Finally/test.jsonl"  # 替换为实际JSONL测试集路径
#
#
#     #base_model_path = "/home/sxk1/EI/model/3-0.6B/Qwen/Qwen3-0.6B"
#     #base_model_path = "/home/sxk1/EI/model/3-1.7B/Qwen/Qwen3-1.7B"
#     #base_model_path = "/home/sxk1/EI/model/3-4B/Qwen/Qwen3-4B"
#     base_model_path = "/home/sxk1/EI/model/3-8B/Qwen/Qwen3-8B" # 替换为基础模型路径
#
#
#     #checkpoint_path = "" # 微调CheckPoint路径，若无则设为None
#     #checkpoint_path = "/home/sxk1/EI/result/CheckPoint/Fraud4/0.6B"
#     #checkpoint_path = "/home/sxk1/EI/result/CheckPoint/Fraud4/1.7B"
#     #checkpoint_path = "/home/sxk1/EI/result/CheckPoint/Fraud4/4B"
#     checkpoint_path = "/home/sxk1/EI/result/CheckPoint/Fraud4/8B"
#
#
#     device = "cuda:4" if torch.cuda.is_available() else "cpu"
#     print(f"使用设备: {device}")
#
#     # 运行评估
#     try:
#         TP, FP, TN, FN, P, R, F1 = evaluate_model(
#             test_data_path,
#             base_model_path,
#             checkpoint_path,
#             device
#         )
#     except Exception as e:
#         print(f"评估过程出错: {e}")
#         import traceback
#
#         traceback.print_exc()









#Hunyuan Test


import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


def load_model(model_dir, checkpoint_path=None, device='cuda'):
    """加载基础模型，并可选择加载微调检查点"""
    print(f"使用设备: {device}")

    # 确定从哪个路径加载模型
    load_path = checkpoint_path if checkpoint_path is not None and os.path.exists(checkpoint_path) else model_dir

    model = AutoModelForCausalLM.from_pretrained(
        load_path,  # 直接使用路径，让Transformers处理加载逻辑
        torch_dtype="auto",
        trust_remote_code=True
    )

    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    return model, tokenizer


def predict(model, tokenizer, prompt, device='cuda', debug=False):
    """模型预测函数"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,  # 减少生成长度以加快速度
        do_sample=False,  # 使用贪婪解码保证一致性
        temperature=0.1,  # 低温度保证输出稳定
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def extract_prediction(model_output):
    """
    从模型输出中提取 is_fraud 的布尔值
    返回: (pred_label, parsed_dict)
    """
    # 方法1: 尝试解析JSON格式
    try:
        start_idx = model_output.find('{')
        end_idx = model_output.rfind('}') + 1
        if start_idx != -1 and end_idx != 0 and end_idx > start_idx:
            json_str = model_output[start_idx:end_idx]
            data = json.loads(json_str)
            if 'is_fraud' in data:
                parsed_dict = {'is_fraud': bool(data['is_fraud'])}
                return bool(data['is_fraud']), parsed_dict
    except:
        pass

    # 方法2: 查找true/false关键词
    model_output_lower = model_output.lower()
    if 'true' in model_output_lower and 'is_fraud' in model_output_lower:
        return True, {'is_fraud': True}
    elif 'false' in model_output_lower and 'is_fraud' in model_output_lower:
        return False, {'is_fraud': False}

    return None, {}


def evaluate_model_detailed(test_file, model, tokenizer, device='cuda', num_detailed_samples=3):
    """
    评估模型并显示详细结果
    """
    # 读取测试数据
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"成功加载 {len(lines)} 条测试数据")
    print("模型加载成功")

    true_labels = []
    pred_labels = []
    detailed_results = []

    # 处理每个样本
    for i, line in enumerate(lines):
        if (i + 1) % max(1, len(lines) // 10) == 0 or (i + 1) == len(lines):
            progress = (i + 1) / len(lines) * 100
            print(f"进度: {i + 1}/{len(lines)} ({progress:.1f}%)")

        try:
            sample = json.loads(line.strip())
            instruction = sample['instruction']
            input_text = sample['input']
            true_label = sample['label']

            prompt = instruction + input_text
            model_output = predict(model, tokenizer, prompt, device, debug=False)
            pred_label, parsed_dict = extract_prediction(model_output)

            result = {
                'sample_id': i + 1,
                'input_text': input_text,
                'true_label': true_label,
                'pred_label': pred_label,
                'model_output': model_output,
                'parsed_dict': parsed_dict,
                'is_correct': pred_label == true_label if pred_label is not None else None
            }
            detailed_results.append(result)

            if pred_label is not None:
                true_labels.append(true_label)
                pred_labels.append(pred_label)

        except Exception as e:
            print(f"样本 {i + 1} 处理错误: {e}")
            continue

    # 计算评估指标
    if true_labels:
        # 计算混淆矩阵
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == True and p == True)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == False and p == True)
        tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == False and p == False)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == True and p == False)

        precision = precision_score(true_labels, pred_labels, pos_label=True, zero_division=0)
        recall = recall_score(true_labels, pred_labels, pos_label=True, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, pos_label=True, zero_division=0)

        # 打印评估结果
        print("\n" + "=" * 80)
        print("评估结果")
        print("=" * 80)
        print(f"TP (True Positive):  {tp}")
        print(f"FP (False Positive): {fp}")
        print(f"TN (True Negative):  {tn}")
        print(f"FN (False Negative): {fn}")
        print("-" * 80)
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall):    {recall:.4f}")
        print(f"F1分数:            {f1:.4f}")

        # 打印前N个样本的详细信息
        print(f"\n{'-' * 80}")
        print(f"前{num_detailed_samples}个样本的详细预测信息")
        print("-" * 80)

        for i in range(min(num_detailed_samples, len(detailed_results))):
            result = detailed_results[i]
            print(f"\n样本 {result['sample_id']}:")
            print(f"输入文本: {result['input_text'][:100]}..." if len(
                result['input_text']) > 100 else f"输入文本: {result['input_text']}")
            print(f"真实标签: {result['true_label']}")
            print(f"预测标签: {result['pred_label']}")
            print(f"预测成功: {result['is_correct']}")

            # 截断过长的模型输出以便显示
            output_preview = result['model_output']
            if len(output_preview) > 200:
                output_preview = output_preview[:200] + "..."
            print(f"原始模型输出: {output_preview}")

            if result['parsed_dict']:
                print(f"解析后的字典: {result['parsed_dict']}")
            else:
                print("解析后的字典: 解析失败")

            print("-" * 40)

    return detailed_results, true_labels, pred_labels


# 主执行函数
if __name__ == "__main__":
    # 配置参数
    test_file = "/home/sxk1/EI/Data/6Loan/Finally/test.jsonl"  # 替换为实际测试文件路径
    model_dir = "/home/sxk1/EI/model/ERNIE/PaddlePaddle/ERNIE-4.5-0.3B-PT"
    checkpoint_path = "/home/sxk1/EI/result/CheckPoint/Fraud6/BRNIE"  # 设为None测试基础模型，或指定检查点路径
    device = "cuda:2"

    # 加载模型
    print("正在加载模型...")
    model, tokenizer = load_model(model_dir, checkpoint_path, device)

    # 评估模型
    print("开始评估...")
    detailed_results, true_labels, pred_labels = evaluate_model_detailed(
        test_file, model, tokenizer, device, num_detailed_samples=3
    )