with import <nixpkgs> {};
mkShell rec {
  buildInputs = [
    pdm
    python312
  ];
  
  NIX_LD_LIBRARY_PATH = lib.makeLibraryPath [
    stdenv.cc.cc
    zlib
    pdm
  ];
  LD_LIBRARY_PATH = NIX_LD_LIBRARY_PATH;
  NIX_LD = lib.fileContents "${stdenv.cc}/nix-support/dynamic-linker";
  shellHook = ''
  source .venv/bin/activate
  '';
}
