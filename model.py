import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass()
class ModelArgs:
    dim: int  = 4096
    n_layers: int = 32
    n_heads: int =32
    n_kv_heads: Optional[int] = None
    vocab_size: int =-1
    multiple_of: int =256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5


     ## Needde for kv cache
    max_batch_size : int=32
    max_seq_len : int =2048

    device:str = None


def precomputer_theta_pos_frequencies(head_dim: int,seq_len:int, device:str,thetha:float = 10000.0):
    assert head_dim%2==0, "Dimension must be divisble by 2"

    thetha_numerator = torch.arange(0,head_dim,2).float()  ### This is the series [0,2,4,head_dim/2]

    ## formule 10000 ^ -(i/head_dim) , where i is [0,2,4,head_dim/2] , shape : head_dim/2
    thetha = 1.0/ (thetha**(thetha_numerator/head_dim)).to(device)

    ### now lets create m (positions)
    ## shape : (seq_len)
    m=torch.arange(0,seq_len).float()


    ## now get all possible combinations of thetha with m (outer product)
    ##shape : (seq_len,head_dim/2)
    freqs=torch.outer(m,thetha)

    ## now write this in complex form
    freqs_complex = torch.polar(torch.ones_like(freqs),freqs)

    return freqs_complex


def apply_rotary_embedding(x:torch.Tensor,freqs_complex:torch.Tensor,device:str):
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self,dim: int,norm_eps: float = 1e-6):
        super().__init__()
        self.eps=norm_eps
        self.weight=nn.Parameter(torch.ones(dim))

    def _norm(self,x:torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

    def forward(self,x:torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight*self._norm(x.float()).type_as(x)



def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class SelfAttention(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()

        ## Indicates the number of heads for the Key and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
       ## Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        ## Indicates how many times the key and values should be repeated
        self.n_rep = self.n_heads_q//self.n_kv_heads
        ## Indicates the dimension of each head, that is the part of embedding that each head will be responsible for
        self.head_dim = args.dim//args.n_heads


        ## Weight for Q
        self.wq=nn.Linear(args.dim,self.n_heads_q*self.head_dim,bias=False)
        ## Weight for K
        self.wk=nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)
        ## Weight for V
        self.wv=nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)

        self.wo=nn.Linear(args.n_heads*self.head_dim,args.dim,bias=False)

        self.cache_k = torch.zeros((args.max_batch_size,args.max_seq_len,self.n_kv_heads,self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self,x:torch.Tensor,start_pos:int,freqs_complex:torch.Tensor):
        batch_size,seq_len, _ = x.shape ## (Batch,1,Dim)

        xq=self.wq(x)
        xk=self.wk(x)
        xv=self.wv(x)

        ## (B,1,H_Q*Head_dim) --> (B,seq_len,H_Q,head_dim)
        xq=xq.view(batch_size,seq_len,self.n_heads_q,self.head_dim)
        ## (B,1,H_Q*Head_dim) --> (B,seq_len,H_Q,head_dim)
        xk=xk.view(batch_size,seq_len,self.n_heads_q,self.head_dim)
        ## (B,1,H_Q*Head_dim) --> (B,seq_len,H_Q,head_dim)
        xv=xv.view(batch_size,seq_len,self.n_heads_q,self.head_dim)

        ## (B, seq_len, H_Q, head_dim) --> (B,seq_len,H_Q,head_dim)
        xq=apply_rotary_embedding(xq,freqs_complex,device=x.device)
        ## (B, seq_len, H_Q, head_dim) --> (B,seq_len,H_Q,head_dim)
        xk=apply_rotary_embedding(xk,freqs_complex,device=x.device)

        ## Replace the  etry in the cache for this token only on K and V
        self.cache_k[:batch_size,start_pos:start_pos+seq_len]=xk
        self.cache_v[:batch_size,start_pos:start_pos+seq_len]=xv

        ## Retrieve all cached values so far because we need it for matmul
        keys=self.cache_k[:batch_size,0:start_pos+seq_len]
        values=self.cache_v[:batch_size,0:start_pos+seq_len]

        ## Since every group Q shares the same K and V heads , just repeast K and V heads for every Q in the same group

        keys=repeat_kv(keys,self.n_rep)
        values=repeat_kv(values,self.n_rep)

        ## move head_dim before seq each head wil watch
        ##(B,1,H_Q,Head_dim) --> (B,H_Q,1,Head_dim)
        xq=xq.transpose(1,2)
        keys=keys.transpose(1,2)
        values=values.transpose(1,2)


        scores=torch.matmul(xq,keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores=F.softmax(scores.float(),dim=-1).type_as(xq)

        output = torch.matmul(scores,values)

        output = (output.transpose(1,2).contiguous().view(batch_size,seq_len,-1))
        return self.wo(output) ##(B,1,Dim) --> (B,1,Dim)





class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x





class EncoderBlock(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        self.n_heads=args.n_heads
        self.dim=args.dim
        self.head_dim=args.dim//args.n_heads

        self.attention=SelfAttention(args)
        self.feed_forward=FeedForward(args)

        ## RMSlayer before self-attention
        self.attention_norm=RMSNorm(args.dim,norm_eps=args.norm_eps)

        ## RMSlayer before feed forward block
        self.ffn_norm=RMSNorm(args.dim,norm_eps=args.norm_eps)

    def forward(self,x:torch.Tensor,start_pos:int,freqs_complex:torch.Tensor):
        ##(B,seq_len,dim) + (B,seq_len,dim) ---> (B,seq_len,dim)
        h=x+self.attention.forward(self.attention_norm(x),start_pos,freqs_complex)
        out=h+self.feed_forward(self.ffn_norm(h))

        return out


class Transformer(nn.Module):
    def __init__(self,args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size!=-1, "vocab size must be set"
        self.args =args
        self.vocab_size=args.vocab_size
        self.n_layers=args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size,args.dim) ## each token is encoded in 4096 dimension
        self.layers=nn.ModuleList()

        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))


        self.norm=RMSNorm(args.dim,norm_eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size,bias=False)  ### output layer each vocab size gets a probability assigned

        self.freqs_complex = precomputer_theta_pos_frequencies(self.args.dim // self.args.n_heads,self.args.max_seq_len*2,device=self.args.device)


    def forward(self,tokens:torch.Tensor, start_pos : int):
        ## (batch_size,seq_len):
        batch_size,seq_len = tokens.shape
        assert seq_len==1 ## only one token at a time is processed"

        ## (b,seq_len) -- > (b,seq_len,dim)
        h = self.tok_embeddings(tokens)

        ## Retrieve the pairs (m,theta) corresponding to the positions [start_pos,start_pos+seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]


        for layer in self.layers:
            h=layer(h,start_pos,freqs_complex)

        h=self.norm(h)

        output=self.output(h).float()
        return output


# freqs=precomputer_theta_pos_frequencies(head_dim=4096/32,seq_len=2048,device='cpu')
#
# x=torch.randn((32,2048,32,128))
#
# x_rot=apply_rotary_embedding(x,freqs,device='cpu')
from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from typing import List

class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int,
              device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        # if device == "cuda":
        #     torch.set_default_tensor_type(torch.cuda.HalfTensor)
        # else:
        #     torch.set_default_tensor_type(torch.BFloat16Tensor)
        if device == "cuda:0":
            torch.set_default_dtype(torch.float16)
            torch.set_default_device("cuda")
        else:
            torch.set_default_dtype(torch.bfloat16)
            torch.set_default_device("cpu")

        model = Transformer(model_args).to(device)

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

        return LLaMA(model, tokenizer, model_args)

    def text_completion(self, prompts: List[str], temperature: float = 0.6, top_p: float = 0.9,
                        max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        # 转换提示词为token
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # 创建 token 张量:
        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.args.device)
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.args.device)

        # 初始化 EOS 标志和提示掩码
        eos_reached = torch.tensor([False] * batch_size, device=self.args.device)
        prompt_tokens_mask = tokens != pad_id  # True if the token is a prompt token, False otherwise
        
        # 生成 token:
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos - 1:cur_pos], cur_pos)
            if temperature > 0:
                # The temperature is applied before the softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedily select the token with the max probability
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace token if it is a padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)

    def _sample_top_p(self, probs, p):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (B, vocab_size)
        # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > p
        # Zero out all the probabilities of tokens that are not selected by the Top P
        probs_sort[mask] = 0.0
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token