# DeepSeek V4.1 Flash for Gufo

A Gufo source extension for DeepSeek V4.1 Flash (`deepseek41`) on Linux x86-64 /
Strix Halo `gfx1151`.

**Status:** in progress. This revision registers provider metadata only; use the
[franzmoca/gufo](https://github.com/franzmoca/gufo) fork to run the model until
the port is complete.

Maintainer: Francesco Moca ([@franzmoca](https://github.com/franzmoca)).

## Build with Gufo

```sh
cmake --preset release -DGUFO_EXTENSION_DIRS=/path/to/deepseek41flash-gufo-extension
cmake --build --preset release --parallel 4
./build/release/gufo diagnose --section extensions
```

With Nix, `nix build` in this repository builds the pinned Gufo revision in
`flake.nix` with this extension.
