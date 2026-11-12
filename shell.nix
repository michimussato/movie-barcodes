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
  nativeBuildInputs = with pkgs; [
    python313
    python313Packages.pip
    imagemagick
  ];
}
