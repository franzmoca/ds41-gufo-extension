{
  description = "Pinned Gufo source-extension package";
  inputs.gufo.url = "github:gufo-org/gufo";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self.outPath ];
    };
    # Gufo's server command generator, reading this package's provider
    # metadata (draft options included).
    lib.x86_64-linux.mkServe = args: gufo.lib.mkGufoServe (args // {
      system = "x86_64-linux";
      gufo = self.packages.x86_64-linux.default;
    });
  };
}
