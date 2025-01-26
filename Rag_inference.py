import json
import time
from typing import List, Tuple
import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
from model import ModelArgs, Transformer, LLaMA

def load_squad_data(file_path: str) -> Tuple[List[str], List[List[str]]]:
    """
    加载 SQuAD 数据集
    :param file_path: SQuAD 数据集的文件路径
    :return: 问题和答案的列表
    """
    with open(file_path, 'r') as f:
        squad_data = json.load(f)
    
    questions = []
    references = []
    
    for item in squad_data['data']:
        for paragraph in item['paragraphs']:
            for qa in paragraph['qas']:
                questions.append(qa['question'])
                if qa['is_impossible']:
                    references.append([""])  # 对于无法回答的问题，参考答案为空
                else:
                    references.append([answer['text'] for answer in qa['answers']])
    
    return questions, references

class Benchmark:
    def __init__(self, model: LLaMA):
        self.model = model

    def benchmark_generation(self, prompts: List[str]) -> Tuple[float, float]:
        """
        测试生成速度和吞吐量
        :param prompts: 提示列表
        :return: 吞吐量 (tokens/s) 和 请求延迟 (req/s)
        """
        start_time = time.time()
        out_tokens, out_texts = self.model.text_completion(prompts, max_gen_len=200)
        end_time = time.time()

        total_tokens = sum(len(tokens) for tokens in out_tokens)
        total_time = end_time - start_time

        throughput = total_tokens / total_time  # tokens/s
        latency = total_time / len(prompts)  # req/s

        return throughput, latency

    def evaluate_generation_quality(self, generated_texts: List[str], reference_texts: List[List[str]]) -> float:
        """
        评估生成文本的质量
        :param generated_texts: 生成的文本列表
        :param reference_texts: 参考文本列表
        :return: 质量评分（BLEU）
        """
        references = [[ref.split() for ref in refs] for refs in reference_texts]
        candidates = [gen.split() for gen in generated_texts]

        score = corpus_bleu(references, candidates)
        return score

if __name__ == '__main__':
    torch.manual_seed(42)

    # 设备设置
    allow_cuda = True
    device = 'cuda:0' 
    # if torch.cuda.is_available() and allow_cuda else 'cpu'

    # 加载 SQuAD 数据集
    squad_file_path = '/home/njh/LLM/dev-v2.0.json'  # 替换为你的 SQuAD 数据集路径
    questions, references = load_squad_data(squad_file_path)

    # 初始化模型
    model = LLaMA.build(
        checkpoints_dir='/home/njh/LLM/Llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=512,
        max_batch_size=len(questions),
        device=device
    )

    print("Model loaded successfully.")

    # 创建基准测试实例
    benchmark = Benchmark(model)

    # 进行基准测试
    throughput, latency = benchmark.benchmark_generation(questions)
    print(f'Throughput: {throughput:.2f} tokens/s')
    print(f'Latency: {latency:.2f} req/s')

    # 生成文本以进行质量评估
    generated_texts = model.text_completion(questions, max_gen_len=200)[1]
    quality_score = benchmark.evaluate_generation_quality(generated_texts, references)
    print(f'Quality Score (BLEU): {quality_score:.4f}')