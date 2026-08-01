#!/usr/bin/env python3
"""Generate the deterministic Layer-0 code_newline oracle checkpoints offline."""
from __future__ import annotations
import argparse, hashlib, json, os, struct, sys
from pathlib import Path
import torch
from safetensors import safe_open
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention, Qwen3MoeMLP, Qwen3MoeRMSNorm, Qwen3MoeTopKRouter
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.reference.prove_minimal_oracle_selective_load import config_from_manifest, read_json
from scripts.validate_minimal_oracle_source import validate

NAMES = ("layer0.post_attention_rmsnorm", "layer0.router_logits", "layer0.selected_expert_ids", "layer0.routing_weights", "layer0.selected_expert_output", "layer0.aggregated_moe_output")
DTYPES = ("F32", "F32", "I64", "F32", "F32", "F32")
SHAPES = ((4,2048),(4,128),(4,8),(4,8),(4,8,2048),(4,2048))
GUARD = [16,114,1,98,84,100,52,53]
TOKENS = [87,28,16,198]
def resolve(root,index,name):
    with safe_open(root/index['weight_map'][name], framework='pt', device='cpu') as f: return f.get_tensor(name)
def assign(module,name,value):
    target=module
    parts=name.split('.')
    for part in parts[:-1]: target=getattr(target,part)
    setattr(target,parts[-1],torch.nn.Parameter(value,requires_grad=False))
def payload(tensors):
    records=[]; blobs=[]; offset=0
    for name,dtype,shape,tensor in zip(NAMES,DTYPES,SHAPES,tensors,strict=True):
        value=tensor.detach().cpu().contiguous()
        raw=value.numpy().astype('<f4' if dtype=='F32' else '<i8',copy=False).tobytes(order='C')
        records.append({'name':name,'dtype':dtype,'shape':list(shape),'offset':offset,'byte_length':len(raw)})
        blobs.append(raw); offset+=len(raw)
    header=json.dumps({'format_version':1,'encoding':'little-endian-contiguous','tensors':records},sort_keys=True,separators=(',',':')).encode()+b'\n'
    return b'CLR-R1-1B\0'+struct.pack('<Q',len(header))+header+b''.join(blobs),records
def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    os.environ.update(TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1')
    torch.manual_seed(20260714);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    validate(a.root,a.manifest); man=read_json(a.manifest); index=read_json(a.root/man['safetensors']['index_file']);cfg=config_from_manifest(man)
    prefix='model.layers.0'; dense=['model.embed_tokens.weight',f'{prefix}.input_layernorm.weight',f'{prefix}.self_attn.q_proj.weight',f'{prefix}.self_attn.k_proj.weight',f'{prefix}.self_attn.v_proj.weight',f'{prefix}.self_attn.o_proj.weight',f'{prefix}.self_attn.q_norm.weight',f'{prefix}.self_attn.k_norm.weight',f'{prefix}.post_attention_layernorm.weight',f'{prefix}.mlp.gate.weight']
    loaded={n:resolve(a.root,index,n).float() for n in dense}; embedding=loaded['model.embed_tokens.weight'][TOKENS].unsqueeze(0)
    with torch.device('meta'):
        norm=Qwen3MoeRMSNorm(2048,eps=1e-6);attn=Qwen3MoeAttention(cfg,0);post=Qwen3MoeRMSNorm(2048,eps=1e-6);router=Qwen3MoeTopKRouter(cfg)
    assign(norm,'weight',loaded[f'{prefix}.input_layernorm.weight']);
    for key in ('q_proj','k_proj','v_proj','o_proj','q_norm','k_norm'): assign(attn,f'{key}.weight',loaded[f'{prefix}.self_attn.{key}.weight'])
    assign(post,'weight',loaded[f'{prefix}.post_attention_layernorm.weight']);assign(router,'weight',loaded[f'{prefix}.mlp.gate.weight'])
    for x in (norm,attn,post,router):x.eval()
    pos=torch.arange(4).unsqueeze(0);mask=torch.zeros((1,1,4,4));mask[0,0].masked_fill_(torch.triu(torch.ones((4,4),dtype=torch.bool),1),torch.finfo(torch.float32).min)
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding
    rotary=Qwen3MoeRotaryEmbedding(cfg,device='cpu');rotary.eval()
    with torch.inference_mode():
        normalized=norm(embedding);attention,_=attn(normalized,rotary(embedding,pos),mask);expert_input=post(embedding+attention);logits,weights,ids=router(expert_input)
        if ids[-1].tolist()!=GUARD: raise RuntimeError(f'router guard mismatch: {ids[-1].tolist()}')
        outs=[]
        for token in range(4):
            token_out=[]
            for expert in ids[token].tolist():
                with torch.device('meta'): mlp=Qwen3MoeMLP(cfg,intermediate_size=768)
                for proj in ('gate','up','down'): assign(mlp,f'{proj}_proj.weight',resolve(a.root,index,f'{prefix}.mlp.experts.{expert}.{proj}_proj.weight').float())
                mlp.eval();token_out.append(mlp(expert_input[:,token,:]).squeeze(0))
            outs.append(torch.stack(token_out))
        selected=torch.stack(outs); aggregate=(selected*weights.unsqueeze(-1)).sum(dim=1)
    raw,records=payload((expert_input.squeeze(0),logits,ids.to(torch.int64),weights,selected,aggregate));a.output.write_bytes(raw)
    print(json.dumps({'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'records':records,'router_guard':ids[-1].tolist(),'versions':{'torch':torch.__version__}},sort_keys=True))
if __name__=='__main__':main()
