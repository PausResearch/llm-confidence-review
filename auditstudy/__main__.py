from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from .common import ROOT, config, load_env, read_json

def main():
    parser=argparse.ArgumentParser(description="Confidence self-audit experiment: prepare -> pilot -> score -> freeze -> evaluation")
    parser.add_argument("--config",default=str(ROOT/"config/study.json"))
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare");p.add_argument("--source",choices=["hf","mock"],default="hf");p.add_argument("--local");p.add_argument("--data-dir")
    p=sub.add_parser("run");p.add_argument("--provider",choices=["deepseek","mock"],default="deepseek");p.add_argument("--phase",choices=["pilot","evaluation"],default="pilot")
    p.add_argument("--stage",choices=["all","generate","review","decide"],default="all");p.add_argument("--limit",type=int,help="Maximum newly scheduled logical tasks; cached tasks do not count")
    p.add_argument("--budget-usd",type=float,help="Lower cumulative spending cap for this invocation (cannot exceed config cap)");p.add_argument("--retry-uncertain",action="store_true",help="Explicitly allow resending an interrupted request without a saved validated response; may incur duplicate billing")
    p.add_argument("--data-dir");p.add_argument("--run-dir")
    for name in ["score","freeze","status"]:
        p=sub.add_parser(name);p.add_argument("--data-dir");p.add_argument("--run-dir",default=str(ROOT/"runs/main"))
        if name=="score":
            p.add_argument("--phase",choices=["pilot","evaluation"],default="pilot");p.add_argument("--bootstrap",type=int)
    sub.add_parser("doctor")
    args=parser.parse_args()
    c=config(args.config)
    load_env()
    if args.command=="doctor":
        import importlib.util
        print(json.dumps({"python":sys.version.split()[0],"project":str(ROOT),"api_key_configured":bool(os.getenv("DEEPSEEK_API_KEY")),
            "model":c["model"],"thinking":False,"budget_usd":c["budget_usd"],"packages":{m:importlib.util.find_spec(m) is not None for m in ["numpy","scipy","pyarrow"]}},indent=2))
        return
    if args.command=="prepare":
        from .data import prepare
        data_dir=Path(args.data_dir or ROOT/("data/mock" if args.source=="mock" else "data/main"))
        prepare(c,data_dir,args.source,args.local)
        return
    if args.command=="run":
        from .runner import run
        mock=args.provider=="mock"
        data_dir=Path(args.data_dir or ROOT/("data/mock" if mock else "data/main"))
        run_dir=Path(args.run_dir or ROOT/("runs/mock" if mock else "runs/main"))
        result=run(c,data_dir,run_dir,args.provider,args.phase,args.stage,args.limit,args.budget_usd,args.retry_uncertain)
        if result["errors"] or result.get("stop_reason"):
            sys.exit(2)
        return
    run_dir=Path(args.run_dir)
    study=read_json(run_dir/"study.json")
    data_dir=Path(args.data_dir or study["data_dir"])
    if args.command=="score":
        from .analysis import score
        score(data_dir,run_dir,args.phase,args.bootstrap or study["config"]["bootstrap_samples"])
    elif args.command=="freeze":
        from .runner import freeze
        freeze(c,data_dir,run_dir)
    else:
        from .store import TaskStore,Ledger
        print(json.dumps({"provider":study["provider"],"data_kind":study["kind"],"tasks":TaskStore(run_dir/"tasks.sqlite").statuses(),
            "frozen":(run_dir/"frozen.json").exists(),"global_spend":Ledger(ROOT/"runs/spend.sqlite",c["budget_usd"]).summary()},indent=2))

if __name__=="__main__":
    try:
        main()
    except (ValueError,FileNotFoundError) as error:
        print("ERROR: "+str(error),file=sys.stderr)
        sys.exit(2)
