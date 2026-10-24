"""Train the SystemOne-Indic dual-encoder from the full IndicJevBench dataset."""
from __future__ import annotations
import argparse
import torch
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, XLMRobertaTokenizer

MODEL = "microsoft/Multilingual-MiniLM-L12-H384"

def rows():
    data = load_dataset("cmul8-hf/IndicJevBench", "default")
    for split in data.values():
        for row in split:
            q = row["question"]
            opts = q.get("options") or (["no", "yes"] if q["type"] == "noul" else [])
            if opts and 0 <= row["expected"] < len(opts):
                yield row["state"], q["instructions"], opts, row["expected"]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--epochs", type=int, default=3); p.add_argument("--batch-size", type=int, default=8); p.add_argument("--max-length", type=int, default=64); p.add_argument("--output", default="artifacts/models/systemone-indic-v1"); a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    tok=XLMRobertaTokenizer.from_pretrained(MODEL); enc=AutoModel.from_pretrained(MODEL).to(device); enc.train()
    data=list(rows())
    def collate(batch): return batch
    loader=DataLoader(data,batch_size=a.batch_size,shuffle=True,collate_fn=collate)
    opt=torch.optim.AdamW(enc.parameters(),lr=2e-5)
    for epoch in range(a.epochs):
        for batch in loader:
            losses=[]
            for state,instruction,options,target in batch:
                x=tok(state,return_tensors="pt",truncation=True,max_length=a.max_length).to(device)
                z=F.normalize(enc(**x).last_hidden_state[:,0],dim=-1)
                choices=[f"{instruction}: {o}" for o in options]
                y=tok(choices,return_tensors="pt",padding=True,truncation=True,max_length=a.max_length).to(device)
                c=F.normalize(enc(**y).last_hidden_state[:,0],dim=-1)
                losses.append(F.cross_entropy((z@c.T)*20,torch.tensor([target],device=device)))
            loss=torch.stack(losses).mean(); opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch={epoch+1} loss={loss.item():.4f}")
    enc.save_pretrained(a.output); tok.save_pretrained(a.output)
if __name__ == "__main__": main()
