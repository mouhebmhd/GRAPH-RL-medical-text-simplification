import os
# default_parts() now loads a Hugging Face NLI model by default (audit item 5). Unit tests must stay offline and fast.
os.environ.setdefault("SIPG_NLI", "off")
