"""Interpreter-startup shim: keep one OpenSSL runtime in the process.

This venv's CPython (uv / python-build-standalone) links OpenSSL 3.5.5
statically. The system's /etc/ssl/openssl.cnf activates a provider section, so
the first OpenSSL init dlopens the system provider modules from
/usr/lib/x86_64-linux-gnu/ossl-modules/, which link the system
libcrypto.so.3 (3.5.7). Both runtimes then live in one process and any later
_hashlib init segfaults -- which took down `import pandas` (pandas 3 imports
pyarrow), and with it sklearn, sentence_transformers and longtracer.

Skipping the config file leaves OpenSSL's built-in default provider active, so
hashing and TLS still work; only the unused pkcs11 provider is not loaded.
setdefault, so an explicit OPENSSL_CONF in the environment still wins.

site imports this after .pth files put src/ on sys.path, i.e. before any import
that could touch OpenSSL.
"""

import os

os.environ.setdefault("OPENSSL_CONF", "/dev/null")
