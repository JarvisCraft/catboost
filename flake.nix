{
  description = "CatBoost - a machine learning method based on gradient boosting over decision trees";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      imports = [ inputs.git-hooks.flakeModule ];
      perSystem =
        {
          system,
          pkgs,
          config,
          lib,
          ...
        }:
        let
          stdenv = pkgs.llvmPackages.stdenv;
          catboostmodel = stdenv.mkDerivation (finalAttrs: {
            pname = "catboostmodel";
            version = "master";

            src = ./.;

            outputs = [
              "out"
              "dev"
            ];

            nativeBuildInputs = with pkgs; [
              cmake
              ninja
              python3
              git
              ragel
              yasm
              llvmPackages.bintools
            ];

            buildInputs = with pkgs; [
              openssl
              zlib
            ];

            env = {
              PROGRAM_VERSION = finalAttrs.version;
              # The generated CMake adds -nodefaultlibs on Linux, so the C and C++
              # standard libraries must be re-added explicitly.
              NIX_LDFLAGS = "-lc -lm";
              # The recursive-library link order (libcxxrt before libcxxabi-parts)
              # only works with a rescanning linker; upstream CI uses lld.
              NIX_CFLAGS_LINK = lib.optionalString stdenv.hostPlatform.isLinux "-fuse-ld=lld";
              # Newer clang turns this warning into a hard error.
              NIX_CFLAGS_COMPILE = toString (
                lib.optionals stdenv.cc.isClang [
                  "-Wno-error=missing-template-arg-list-after-template-kw"
                ]
              );
            };

            postPatch = ''
              # Conan would provide ragel/yasm in $BUILD/bin; point the generator
              # macros at the nixpkgs binaries instead.
              substituteInPlace cmake/common.cmake \
                --replace-fail "\''${RAGEL_BIN}" "${pkgs.ragel}/bin/ragel" \
                --replace-fail "\''${YASM_BIN}" "${pkgs.yasm}/bin/yasm"
              # The generated CMake references Conan's target name `openssl::openssl`;
              # alias it to the FindOpenSSL target.
              for cmakelists in $(find . -name "CMakeLists.*"); do
                sed -i "s/openssl::openssl/OpenSSL::SSL/g" "$cmakelists"
              done
            '';

            # nixpkgs' cmake setup hook provides the configure phase (out-of-source
            # in build/, -GNinja because ninja provides the build phase) and ninja's
            # hook the build phase; keep the install phase manual so only the C API
            # artifacts end up in $out.
            cmakeFlags = [
              (lib.cmakeFeature "CATBOOST_COMPONENTS" "libs")
              (lib.cmakeBool "CMAKE_POSITION_INDEPENDENT_CODE" true)
            ];

            ninjaFlags = [
              "catboostmodel"
              "catboostmodel_static"
            ];

            installPhase = ''
              runHook preInstall

              mkdir -p "$out/lib" "$dev/include"
              install -m755 "catboost/libs/model_interface/libcatboostmodel${stdenv.hostPlatform.extensions.sharedLibrary}" "$out/lib/"
              install -m644 "catboost/libs/model_interface/static/libcatboostmodel_static.a" "$out/lib/"
              install -m644 "../catboost/libs/model_interface/c_api.h" "$dev/include/"

              runHook postInstall
            '';

            meta = {
              description = "C API of CatBoost, a gradient boosting on decision trees library";
              homepage = "https://catboost.ai";
              license = lib.licenses.asl20;
              platforms = lib.platforms.unix;
            };
          });
        in
        {
          _module.args.pkgs = import inputs.nixpkgs { inherit system; };
          formatter = pkgs.nixfmt;
          packages.default = catboostmodel;
          devShells.default = pkgs.mkShell {
            __structuredAttrs = true;
            strictDeps = true;
            inputsFrom = [ catboostmodel ];
            shellHook = config.pre-commit.installationScript;
            packages = with pkgs; [
              pkg-config
              gdb
              valgrind
              clang
              clang-tools
            ];
          };
          pre-commit.settings = {
            package = pkgs.prek;
            hooks = {
              nixfmt = {
                enable = true;
                # Vendored Cython's override.nix is not nixfmt-formatted; the hook
                # would rewrite it in place and fail the commit if it is staged.
                excludes = [ "contrib/tools/cython/" ];
              };
              nil.enable = true;
              statix.enable = true;
              flake-checker.enable = true;
              deadnix.enable = true;
            };
          };
        };
    };
}
