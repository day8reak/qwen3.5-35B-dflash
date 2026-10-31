# Tokenizer dependencies

These sources are built statically into the optional text frontend of the mixed
resident supervisor. No library is installed into the SDK or board system image.

* Oniguruma 6.9.10: https://github.com/kkos/oniguruma/tree/v6.9.10
  (`oniguruma/COPYING`). This is the regex engine also used by the default
  Hugging Face tokenizers build; the model's literal pattern is retained.
* utf8proc 2.11.0: https://github.com/JuliaStrings/utf8proc/tree/v2.11.0
  (`utf8proc/LICENSE.md`). Used for UTF-8 validation and NFC normalization.

Only required source/build files are retained. `sources.lock.json` records the
download archive SHA256 and each retained upstream file. Changes to upstream
sources require updating that provenance; application logic lives outside these
directories. CMake generates compiler/platform configuration in the run-owned
build directory. On-device tokenizer behavior is checked against the host's
original tokenizer IDs before any model load.
