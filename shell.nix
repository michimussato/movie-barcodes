{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  LD_LIBRARY_PATH = with pkgs; lib.makeLibraryPath [
    stdenv.cc.cc
    libz
    libxcb
    libGL
    glib
  ];
}
