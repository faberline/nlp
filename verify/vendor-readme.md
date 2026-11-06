# Faberline NLP CNL parser

This copy comes from the cnl directory of Faberline NLP at commit
e9bb5cbd8a8bb7d9fac12a4d455ebfc0761bc24c. SOM can build with this copy in a
standalone checkout.

```text
d8aca3171370f0296f2899b55240858fc97de50253c19ed9f749e5fbab781f67  Cargo.toml
84abaab7d923faea3c88c4d7d6c761752af722be94407ca59c9abc5e680fab8e  src/lib.rs
8c13d0f73b3ce9daa63688a9072aff0ee618c4205e285ca5a309c543baba418b  src/som_cnl.pest
```

## Responsibilities

- NLP implements the English grammar and reports positions in the original
  text. The parser can return a syntax tree or check the syntax alone.
- SOM owns the CNL contract, including supported versions and business
  rules. SOM turns the input into a typed prompt, then plans the DSL and
  generates files. SOM also owns the public error codes and error order.
- SOM uses the NLP grammar for its first syntax check. The original syntax
  check preserves existing inputs and diagnostics during the transition.

## Updates

Copy the Cargo manifest and the two source files from one merged NLP
commit. Check that the three files match that commit byte for byte. SOM
includes them in the compiler fingerprint.

Run the original SOM test suites on GitHub runners. Compare the typed
prompt, DSL, file paths and file bytes for the same inputs on the old and
new compilers. Compare error codes, positions and order for invalid
inputs. The compiler fingerprint is part of the bundle. Build a fresh
bundle for the new compiler.

Remove the original syntax check only after the NLP path covers all
supported syntax and preserves the output and diagnostic contract on its
own.
