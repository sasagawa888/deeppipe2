defmodule Deeppipe2.MixProject do
  use Mix.Project

  def project do
    [
      app: :deeppipe2,
      version: "1.1.1",
      elixir: "~> 1.7",
      description: "Deep-Learning library with CUDA/CUBLAS",
      compilers: [:makecuda] ++ Mix.compilers(),
      deps: deps(),

      # Docs
      name: "DeepPipe2",
      source_url: "https://github.com/sasagawa888/deeppipe2",
      start_permanent: Mix.env() == :prod,
      docs: [
        # The main page in the docs
        # main: "DeepPipe2",
        extras: ["README.md"]
      ],
      package: [
        files: [
          "lib",
          "makefile",
          "README.md",
          "mix.exs"
        ],
        maintainers: ["Kenichi Sasagawa"],
        licenses: ["modified BSD"],
        links: %{"GitHub" => "https://github.com/sasagawa888/deeppipe2"}
      ]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:matrex, "~> 0.6"},
      {:ex_doc, "~> 0.21", only: :dev, runtime: false}
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
    ]
  end
end

defmodule Mix.Tasks.Compile.Makecuda do
  use Mix.Task

  def run(_) do
    Mix.shell().cmd("make")
    :ok
  end
end
