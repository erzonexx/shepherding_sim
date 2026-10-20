"""Prepare fixed-only prefixes with Spark's cached tokenizer, using CPU only.

No inference service is started and no sheep state or credentials are transferred.
The image/model/chat template must match the inference service. Regenerate after
changing any of them or the instructions. Enable prefix caching and the tested
--prefix-match-unit 16 on the server; this client feature cannot enable server KV
caching, guarantee hits after eviction, or remember changing sheep observations.
"""
import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from vllm_backend import VLLMBackend


def instruction_variants(model):
    rows = []
    for input_format, compact in product(("json", "tables", "dedup"), (False, True)):
        for mode, target in ((True, True), (True, False), (False, True)):
            settings = SimpleNamespace(**{
                key: value for key, value in vars(config).items() if key.isupper()
            })
            settings.VLLM_PREFIX_PRIMING = False
            settings.VLLM_INPUT_FORMAT = input_format
            settings.LLM_CACHE_ENABLED = False
            settings.HIGH_LEVEL_MODE_SELECTION_ENABLED = mode
            settings.HIGH_LEVEL_TARGET_SELECTION_ENABLED = target
            backend = VLLMBackend(settings=settings, model=model, client=object(), compact_output=compact)
            try:
                rows.append({"instructions": backend.instructions,
                             "key": hashlib.sha256(backend.instructions.encode()).hexdigest()})
            finally:
                backend.close()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default="spark_csslab")
    parser.add_argument("--image", default="vllm/vllm-openai:qwen38")
    parser.add_argument("--model", default=config.VLLM_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    # Token offsets, not a text-prefix guess: no token may overlap user content.
    remote = '''import hashlib,json
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
prefixes={}
for row in ROWS:
    fixed=None
    for user in ('SHEPHERD_PREFIX_BOUNDARY_ONE','different boundary two'):
        messages=[{'role':'system','content':row['instructions']},{'role':'user','content':user}]
        rendered=t.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        start=rendered.rindex(user)
        encoded=t(rendered,add_special_tokens=False,return_offsets_mapping=True)
        ids=encoded['input_ids']
        actual=t.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        actual=actual['input_ids'] if hasattr(actual,'keys') else actual
        assert ids==actual
        cut=next(i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>start)
        prefix=ids[:cut]
        assert fixed is None or prefix==fixed
        fixed=prefix
    prefixes[row['key']]={'prompt_ids':fixed,'sha256':hashlib.sha256(json.dumps(fixed).encode()).hexdigest()}
print(json.dumps({'version':1,'model':MODEL,'enable_thinking':False,'includes_dynamic_content':False,
 'chat_template_sha256':hashlib.sha256(t.chat_template.encode()).hexdigest(),'prefixes':prefixes}))
'''.replace('MODEL', repr(args.model)).replace('ROWS', repr(instruction_variants(args.model)))
    command = ['ssh', '-4', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', args.ssh_host,
               'docker', 'run', '--rm', '--pull=never', '--network', 'none', '--memory', '4g',
               '--memory-swap', '4g', '--cpus', '2', '-e', 'NVIDIA_VISIBLE_DEVICES=void',
               '-e', 'CUDA_VISIBLE_DEVICES=', '--mount',
               'type=bind,source=/home/csslab/.cache/huggingface/hub,target=/root/.cache/huggingface/hub,readonly',
               '--entrypoint', 'python3', '-i', args.image, '-']
    result = subprocess.run(command, input=remote, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise SystemExit('CPU prefix preparation failed; check the SSH connection, cached model and image.')
    plan = json.loads(result.stdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'output':str(args.output),'variants':len(plan['prefixes']),
                      'fixed_token_counts':[len(x['prompt_ids']) for x in plan['prefixes'].values()],
                      'dynamic_data_sent':False,'gpu_model_started':False}))


if __name__ == '__main__':
    main()
